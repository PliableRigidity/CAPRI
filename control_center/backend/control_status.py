"""What the ELRS & Control screen is allowed to say. Pure Python: no Qt, no clock, no port, nothing sent.

Everything is derived from what the hub already knows (the listen-only ``ControllerMonitor``, the policy
state) and from files on disk (the wheels-raised safety record). Each item carries its source and its age.
Anything the baseline firmware does not report is shown as NOT AVAILABLE with the reason; no number is ever
invented. Evidence words follow the repo: HARDWARE / MOCK / FIXTURE / SIM / PROPOSED / BLOCKED.

The two user-facing operating modes are MANUAL (RadioMaster over ELRS, independent of this computer) and
AUTONOMOUS (the policy produces the commands; DISABLED in this build). AI SHADOW is a non-actuating test
display. PC-originated commands exist only as an implementation detail of AUTONOMOUS and of the supervised
motion check; they are not a mode.

Nothing in this module writes to the robot. It does not import ``serial``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from control_center.backend.controller_link import ControllerMonitor, ControllerStatus
from control_center.backend.readiness import FAILED, NOT_CONFIGURED, READY, WARNING, Readiness
from control_center.backend import status as st

ROOT = Path(__file__).resolve().parents[2]
MOTION_CHECK_SCRIPT = ROOT / "scripts" / "motion_check.py"

BASELINE_FIRMWARE = "autobot-handoff-bench-1"
WIFI_FIRMWARE = "autobot-handoff-wifi-1"
KNOWN_FIRMWARE = {
    BASELINE_FIRMWARE: "ELRS-Handoff-Bench: the provisional competition baseline",
    WIFI_FIRMWARE: "ELRS-Handoff-WiFi: route D, experimental (restarted and dropped on battery on 20 Nov)",
}

CH_STALE_AFTER_S = 3.0            # CH is sent 4 times a second while disarmed and once a second while armed
RC_FRAME_TIMEOUT_MS = 250         # the firmware's radio frame timeout (README: "the 250 ms frame timeout")
PC_COMMAND_TIMEOUT_MS = 250       # the firmware's stale PC command stop
PC_SENDING_WITHIN_MS = 500        # a PC command this recent means a PC client is live
PC_LOG_RECENT_S = 1.0
RESTART_DROP_MS = 100             # uptime must fall by at least this to count as a restart
RESTART_FAILED_WINDOW_S = 600.0   # a restart this recent keeps the section FAILED; older ones are a WARNING
SAFETY_MAX_AGE_S = 12 * 3600.0    # same limit as scripts/motion_check.py

NOT_SENT_TEXT = "Not available: the firmware does not send it"
POWER_TEXT = ("no measurement: the firmware has no voltage sensing; use the multimeter checks PWR-1/PWR-2 "
              "in the playbook")

HARDWARE, MOCK, FIXTURE, SIM, PROPOSED, BLOCKED, REJECTED = ("HARDWARE", "MOCK", "FIXTURE", "SIM", "PROPOSED",
                                                             "BLOCKED", "REJECTED")

# The current repository state, stated once. Changing either needs a hardware result, not a code edit alone.
COMMAND_PATH_VERIFIED = False
OBSERVATION_PIPELINE_VERIFIED = False
# Even if every prerequisite were met, this build has no GUI action that enables autonomous mode.
AUTONOMOUS_ENABLEABLE_IN_THIS_BUILD = False

AUTONOMOUS_CONFIRMATIONS = {
    "wheels_raised_or_clear": "Wheels raised, or the floor is clear all round",
    "power_cutoff": "The power cut-off is in reach",
    "radiomaster": "The RadioMaster is in my hand and switched on",
}

RETURN_TO_MANUAL = ("Return to MANUAL is always available: set MODE to RADIO, or switch ARM off, on the RadioMaster. "
                    "Neither depends on this computer.")
MANUAL_TEXT = ("The RadioMaster works without this computer. If this window or the PC crashes, manual ELRS driving "
               "is unaffected: the Command Centre is listen-only on the controller link.")
HANDOFF_RULES = (
    "Only one source drives at a time. The controller (firmware) chooses it from the MODE switch; this window cannot "
    "select, arm or change it.",
    "Every switch passes through 'both stopped, neutral confirmed': the firmware brakes, forgets the old command and "
    "changes its token, and the new source must then show neutral (sticks at zero, or a 0,0 command with the new "
    "token) that arrives after the switch. A held stick or key does nothing until released.",
    "Communication timeout (firmware behaviour): no PC command for 250 ms stops the motors and latches PC_STALE; no "
    "valid radio frame for 250 ms (or a fresh 0 % link quality) stops them and latches RADIO_LOST. Separately, this "
    "window treats a controller status older than 0.6 s as STALE (unconfirmed). That is a display rule, not a motor rule.",
    "Failsafe (firmware behaviour): radio loss or any latched fault stops the motors; motion resumes only after ARM "
    "off, then on, on the RadioMaster. There is no automatic fall-back to the other source.",
    RETURN_TO_MANUAL,
)

LEVELS = ("ok", "warn", "bad", "off")


# ---------------------------------------------------------------- motion_check access (lazy, pure)


def motion_check_module():
    """``scripts/motion_check.py`` (pure functions only are used here). None when it cannot be imported."""
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        import motion_check
    except Exception:            # an import problem must close the gates, never open them
        return None
    return motion_check


def _required_checks() -> tuple[str, ...]:
    module = motion_check_module()
    return tuple(module.REQUIRED_CHECKS) if module else ("stop", "disarm", "takeover", "pc_link_loss", "radio_loss")


def _power_choices() -> tuple[str, ...]:
    module = motion_check_module()
    return tuple(module.POWER_CHOICES) if module else ("usb-only", "battery-and-usb", "battery-only")


# ---------------------------------------------------------------- items


@dataclass(frozen=True)
class Item:
    """One displayed fact: where it came from, how old it is, or why it is not available."""

    label: str
    text: str
    level: str = "off"                  # ok / warn / bad / off
    source: str = ""
    age_s: float | None = None
    available: bool = True
    detail: str = ""

    def source_line(self) -> str:
        parts = [self.source] if self.source else []
        if self.available and self.age_s is not None:
            parts.append(age_text(self.age_s))
        return " · ".join(parts)


def age_text(age_s: float | None) -> str:
    if age_s is None:
        return "age unknown"
    return f"{max(0.0, age_s):.1f} s ago"


def not_available(label: str, why: str, source: str = "") -> Item:
    return Item(label, f"NOT AVAILABLE: {why}", "off", source, None, False)


def not_sent(label: str) -> Item:
    return Item(label, NOT_SENT_TEXT, "off", "baseline firmware has no such field", None, False,
                "ELRS-Handoff-Bench sends only HELLO, ST and CH lines (README, Diagnostics).")


def _int(fields: dict[str, str], key: str) -> int | None:
    try:
        return int(fields[key])
    except (KeyError, ValueError):
        return None


# ---------------------------------------------------------------- inputs and the restart tracker


@dataclass
class ControlInputs:
    monitor: ControllerMonitor
    link_label: str | None = None
    link_error: str | None = None
    mock: bool = False
    policy: dict[str, Any] = field(default_factory=dict)
    live_shadow_missing: list[str] | None = None


def inputs_from_hub(hub: Any) -> ControlInputs:
    """Read the hub with safe fallbacks: a missing attribute becomes 'not known', never an error."""
    options = getattr(hub, "options", None)
    missing = None
    getter = getattr(hub, "live_shadow_missing", None)
    if callable(getter):
        try:
            missing = [str(item) for item in getter()]
        except Exception:
            missing = None
    return ControlInputs(monitor=getattr(hub, "monitor", None) or ControllerMonitor(),
                         link_label=getattr(hub, "link_label", None), link_error=getattr(hub, "link_error", None),
                         mock=bool(getattr(options, "mock", False)), policy=dict(getattr(hub, "policy", None) or {}),
                         live_shadow_missing=missing)


class ControlTracker:
    """Remembers what a single look at the monitor cannot show: restarts and how old the CH / HELLO lines are.

    Call ``observe(monitor, now)`` regularly (about 4 times a second). A restart is the controller's own
    uptime (``t=`` of the ST line) going backwards.
    """

    def __init__(self) -> None:
        self.restarts = 0
        self.last_restart_at: float | None = None
        self.last_restart_from_ms: int | None = None
        self.last_restart_to_ms: int | None = None
        self.source_changes = 0
        self.channels_at: float | None = None       # when a changed CH line was first noticed (poll resolution)
        self.hello_at: float | None = None
        self._monitor: ControllerMonitor | None = None
        self._status_obj: Any = None
        self._channels_obj: Any = None
        self._hello_obj: Any = None
        self._last_uptime_ms: int | None = None

    def observe(self, monitor: ControllerMonitor, now: float) -> None:
        if monitor is not self._monitor:
            if self._monitor is not None:
                self.source_changes += 1
            self._monitor = monitor
            # Whatever the new monitor already holds arrived before we looked: its age is unknown, and its
            # uptime is only a baseline (a different connection is not evidence of a restart by itself).
            self._status_obj = monitor.status
            self._last_uptime_ms = monitor.status.device_ms if monitor.status is not None else None
            self._channels_obj, self.channels_at = monitor.channels, None
            self._hello_obj, self.hello_at = monitor.hello, None
            return
        status = monitor.status
        if status is not None and status is not self._status_obj:
            self._status_obj = status
            if self._last_uptime_ms is not None and status.device_ms + RESTART_DROP_MS <= self._last_uptime_ms:
                self.restarts += 1
                self.last_restart_at = now
                self.last_restart_from_ms, self.last_restart_to_ms = self._last_uptime_ms, status.device_ms
            self._last_uptime_ms = status.device_ms
        if monitor.channels is not self._channels_obj:
            self._channels_obj = monitor.channels
            if monitor.channels:
                self.channels_at = now
        if monitor.hello is not self._hello_obj:
            self._hello_obj = monitor.hello
            if monitor.hello:
                self.hello_at = now

    def recent_restart(self, now: float) -> bool:
        return self.restarts > 0 and self.last_restart_at is not None and now - self.last_restart_at <= RESTART_FAILED_WINDOW_S


# ---------------------------------------------------------------- authority banner


@dataclass(frozen=True)
class AuthorityView:
    banner: Item                     # the big word
    modes: tuple[Item, Item, Item]   # MANUAL, AUTONOMOUS, SHADOW
    controller_line: Item            # what the firmware itself says, verbatim
    manual_note: str = MANUAL_TEXT


def _shadow_item(policy: dict[str, Any], mock: bool) -> Item:
    mode = policy.get("mode", "OFF")
    if mode == "LIVE SHADOW":
        return Item("SHADOW", "ACTIVE · non-actuating, display only", "warn", "hub policy state",
                    detail="Proposed actions are shown and never sent to the robot.")
    if mode == "SIM":
        return Item("SHADOW", "ACTIVE · non-actuating, display only (simulator input)", "warn", "hub policy state",
                    detail="The policy reads simulator observations. Its output is never sent to the robot.")
    return Item("SHADOW", "off · non-actuating when on; independent of the motors", "off", "hub policy state")


def authority_view(inputs: ControlInputs, now: float) -> AuthorityView:
    authority = inputs.monitor.authority(now)
    prefix = "MOCK · " if inputs.mock else ""
    source = "MOCK controller replay" if inputs.mock else "controller ST line"
    autonomous = Item("AUTONOMOUS", "DISABLED · not enableable in this build", "off", "autonomous gate",
                      detail="See the AUTONOMOUS MODE box for each unmet prerequisite.")
    shadow = _shadow_item(inputs.policy, inputs.mock)
    if inputs.link_label is None:
        banner = Item("Operating mode", prefix + "UNKNOWN", "off", "no controller telemetry source connected",
                      detail="Connect the controller in SETUP > Controller & IMU. The RadioMaster still drives.")
        manual = Item("MANUAL", "unconfirmed · no controller telemetry", "off", "no source")
        line = not_available("Controller reports", "no controller telemetry source connected")
        return AuthorityView(banner, (manual, autonomous, shadow), line)
    if authority.freshness != "FRESH":
        word = authority.freshness
        banner = Item("Operating mode", prefix + word, "bad" if word == "STALE" else "warn", source, authority.age_s,
                      detail="Who is driving is not confirmed. The old value is not shown as current.")
        manual = Item("MANUAL", f"unconfirmed · telemetry {word}", "warn", source, authority.age_s)
        line = not_available("Controller reports", f"telemetry is {word}; the last report is not shown as current", source)
        return AuthorityView(banner, (manual, autonomous, shadow), line)
    s = authority.status
    age = authority.age_s
    line = Item("Controller reports",
                f"source={s.active} requested={s.requested} armed={int(s.armed)} link={int(s.link)} block={s.block} "
                f"fault={s.fault}", "ok" if s.active in ("RADIO", "NONE") else "warn", source, age,
                detail="This is the firmware's own report (ST line), not a button state.")
    if s.active == "RADIO":
        banner = Item("Operating mode", prefix + "MANUAL", "ok", source, age,
                      detail="The RadioMaster is driving.")
        manual = Item("MANUAL", "ACTIVE · RadioMaster driving", "ok", source, age)
    elif s.active == "PC":
        banner = Item("Operating mode", prefix + "PC-ORIGINATED COMMANDS ACTIVE", "warn", source, age,
                      detail="Something is sending PC commands (a supervised motion check, a diagnostic client, or "
                             "a script). This is not the AUTONOMOUS mode, which is disabled in this build. To return "
                             "to MANUAL: " + RETURN_TO_MANUAL)
        manual = Item("MANUAL", "not driving · PC commands are active", "warn", source, age)
    else:
        banner = Item("Operating mode", prefix + "NOT DRIVING", "warn", source, age,
                      detail=f"Nobody is driving: {s.block}")
        manual = Item("MANUAL", ("selected" if s.requested == "RADIO" else "not selected") + f" · not driving ({s.block})",
                      "ok" if s.requested == "RADIO" and s.fault == "NONE" else "warn", source, age)
    return AuthorityView(banner, (manual, autonomous, shadow), line)


# ---------------------------------------------------------------- handoff state


@dataclass(frozen=True)
class HandoffState:
    phase: str                       # MANUAL / PC_COMMANDS / STOPPED_WAITING_NEUTRAL / STOPPED_FAILSAFE / STOPPED_IDLE / UNCONFIRMED / CONFLICT
    source: str | None               # the single current command source, or None when there is none or it is unknown
    headline: str
    level: str
    conflicts: tuple[str, ...] = ()
    transition: str = ""
    rules: tuple[str, ...] = HANDOFF_RULES
    return_to_manual: str = RETURN_TO_MANUAL


def find_conflicts(status: ControllerStatus, last_pc_command_age_s: float | None) -> list[str]:
    """Every state in which two sources could be live at once. Pure: the inputs are the firmware's own report."""
    found: list[str] = []
    s = status
    if s.requested in ("RADIO", "PC") and s.active in ("RADIO", "PC") and s.requested != s.active:
        found.append(f"the controller reports source={s.active} while the MODE switch asks for {s.requested}: two sources are involved")
    pc_recent = 0 <= s.pc_age_ms <= PC_SENDING_WITHIN_MS
    if pc_recent and (s.requested == "RADIO" or s.active == "RADIO"):
        found.append(f"a PC client is sending commands (pc_age={s.pc_age_ms} ms) while RADIO is selected: the firmware "
                     "should ignore them, but two sources are live")
    if last_pc_command_age_s is not None and last_pc_command_age_s <= PC_LOG_RECENT_S and (s.requested == "RADIO" or s.active == "RADIO") and not pc_recent:
        found.append(f"the client log shows a drive command sent {last_pc_command_age_s:.1f} s ago while RADIO is selected")
    if s.active == "PC" and s.pc_age_ms < 0:
        found.append("the controller reports PC as the source but has received no PC command since boot")
    if s.active != "NONE" and not s.link:
        found.append(f"the controller reports source={s.active} while it reports the radio link down")
    return found


def handoff_state(inputs: ControlInputs, now: float) -> HandoffState:
    monitor = inputs.monitor
    authority = monitor.authority(now)
    if inputs.link_label is None or authority.freshness != "FRESH":
        why = ("no controller telemetry source is connected" if inputs.link_label is None
               else f"controller telemetry is {authority.freshness}")
        return HandoffState("UNCONFIRMED", None, f"Source UNCONFIRMED: {why}. Either source may be live; the firmware's own "
                            "250 ms stops still apply and the RadioMaster still has the last word.", "warn")
    s = authority.status
    pc_age = None if monitor.last_pc_command_at is None else max(0.0, now - monitor.last_pc_command_at)
    conflicts = find_conflicts(s, pc_age)
    transition = ""
    if s.requested != s.active and s.requested in ("RADIO", "PC"):
        transition = f"Hand-over in progress or blocked: the RadioMaster asks for {s.requested}, the controller confirms {s.active} ({s.block})."
    if conflicts:
        return HandoffState("CONFLICT", None, "CONFLICT: " + "; ".join(conflicts), "bad", tuple(conflicts), transition)
    if s.fault != "NONE" or not s.link:
        cause = s.fault if s.fault != "NONE" else s.block
        return HandoffState("STOPPED_FAILSAFE", None,
                            f"STOPPED (failsafe): {cause}. Motion needs ARM off, then on, on the RadioMaster.", "bad",
                            (), transition)
    if s.active == "RADIO":
        return HandoffState("MANUAL", "RADIO", "Single command source: RADIO (the RadioMaster). Nothing else is driving.", "ok", (), transition)
    if s.active == "PC":
        return HandoffState("PC_COMMANDS", "PC", "Single command source: PC-originated commands. This build does not "
                            "start them; a supervised motion check or another client did.", "warn", (), transition)
    if s.block in ("WAIT_RADIO_NEUTRAL", "WAIT_PC_NEUTRAL"):
        wanted = "RADIO" if s.block == "WAIT_RADIO_NEUTRAL" else "PC"
        return HandoffState("STOPPED_WAITING_NEUTRAL", None, f"Both sources stopped: waiting for neutral from {wanted} ({s.block}).",
                            "warn", (), transition)
    return HandoffState("STOPPED_IDLE", None, f"No source is driving: {s.block}.", "warn" if s.block != "DISARMED" else "off", (), transition)


# ---------------------------------------------------------------- ELRS, firmware, power, communication


def _ch_state(inputs: ControlInputs, tracker: ControlTracker, now: float) -> tuple[dict[str, str] | None, float | None, str]:
    """(CH fields, age, why-not). The CH line has no timestamp of its own: its age is when a change was noticed."""
    channels = inputs.monitor.channels
    if not channels:
        return None, None, "no CH line received"
    if tracker.channels_at is None:
        return None, None, "a CH line was received before this window began watching and none since: its age is unknown"
    age = now - tracker.channels_at
    if age > CH_STALE_AFTER_S:
        return None, age, f"the last CH line is {age:.1f} s old (it is sent at least once a second)"
    return channels, age, ""


def elrs_items(inputs: ControlInputs, tracker: ControlTracker, now: float) -> list[Item]:
    authority = inputs.monitor.authority(now)
    fresh = authority.fresh and inputs.link_label is not None
    s = authority.status if fresh else None
    st_src = "MOCK controller replay" if inputs.mock else "controller ST line"
    ch_src = "MOCK controller replay" if inputs.mock else "controller CH line"
    why_st = ("no controller telemetry source connected" if inputs.link_label is None
              else f"controller telemetry is {authority.freshness}")
    items: list[Item] = []

    if s is None:
        items.append(not_available("ELRS link (ST link)", why_st, st_src))
    else:
        items.append(Item("ELRS link (ST link)", "UP · qualified radio link" if s.link else "DOWN · no qualified radio link",
                          "ok" if s.link else "bad", st_src, authority.age_s,
                          detail="link=1 means fresh CRC-valid RC frames, in a row, and no fresh 0 % link quality."))
    channels, ch_age, ch_why = _ch_state(inputs, tracker, now)
    if channels is None:
        for label in ("Link quality (CH lq)", "Newest RC frame age (CH rc_age)", "RC frame counters (CH)"):
            items.append(not_available(label, ch_why, ch_src))
    else:
        lq, rc_age = _int(channels, "lq"), _int(channels, "rc_age")
        if lq is None:
            items.append(not_available("Link quality (CH lq)", "the CH line has no readable lq field", ch_src))
        elif lq < 0:
            items.append(not_available("Link quality (CH lq)", "the receiver has not sent link statistics since boot (lq=-1)", ch_src))
        else:
            items.append(Item("Link quality (CH lq)", f"{lq} %", "bad" if lq == 0 else "ok", ch_src, ch_age,
                              detail="The last value the receiver reported; the firmware does not say how old it is. 0 % counts as radio lost."))
        if rc_age is None:
            items.append(not_available("Newest RC frame age (CH rc_age)", "the CH line has no readable rc_age field", ch_src))
        elif rc_age < 0:
            items.append(not_available("Newest RC frame age (CH rc_age)", "no valid RC frame since boot", ch_src))
        else:
            late = rc_age >= RC_FRAME_TIMEOUT_MS
            items.append(Item("Newest RC frame age (CH rc_age)", f"{rc_age} ms" + (" · beyond the firmware's 250 ms timeout" if late else ""),
                              "bad" if late else "ok", ch_src, ch_age))
        frames, crc, length, pc_bad = (_int(channels, key) for key in ("frames", "crc", "len", "pc_bad"))
        if None in (frames, crc, length):
            items.append(not_available("RC frame counters (CH)", "the CH line has no readable frames/crc/len fields", ch_src))
        else:
            items.append(Item("RC frame counters (CH)", f"{frames} good · {crc} bad CRC · {length} wrong length",
                              "warn" if crc or length else "ok", ch_src, ch_age,
                              detail="Counts since the controller booted; a rate is not computed."))

    # Receiver state: only from fields that exist.
    note = "derived from ST link/block and CH rc_age; the receiver itself reports no state"
    if s is None:
        items.append(not_available("Receiver state (derived)", why_st, note))
    elif not s.link:
        text = {"NO_RADIO": "no qualified radio link since boot (block=NO_RADIO)",
                "RADIO_LOST": "radio link lost (block=RADIO_LOST)"}.get(s.block, f"no qualified link (block={s.block})")
        items.append(Item("Receiver state (derived)", text, "bad", note, authority.age_s))
    elif channels is not None and _int(channels, "rc_age") is not None and 0 <= _int(channels, "rc_age") < RC_FRAME_TIMEOUT_MS:
        items.append(Item("Receiver state (derived)", f"delivering valid RC frames (rc_age {_int(channels, 'rc_age')} ms)", "ok", note, ch_age))
    else:
        items.append(Item("Receiver state (derived)", "link qualified per ST; frame age not confirmed (no current CH line)", "warn", note, authority.age_s))

    # Failsafe / radio loss from fault, stop and block.
    if s is None:
        items.append(not_available("Failsafe / radio loss (ST fault, stop, block)", why_st, st_src))
    elif s.fault != "NONE" or s.block in ("RADIO_LOST", "REARM_REQUIRED"):
        radio = "RADIO_LOST" in (s.fault, s.block, s.stop)
        items.append(Item("Failsafe / radio loss (ST fault, stop, block)",
                          (f"{'RADIO LOST' if radio else 'FAULT LATCHED'} · fault={s.fault} block={s.block} stop={s.stop} · "
                           "stopped; needs ARM off, then on, on the RadioMaster"), "bad", st_src, authority.age_s,
                          detail="Firmware behaviour: radio loss or a latched fault stops the motors and latches until ARM is cycled."))
    elif s.block == "NO_RADIO":
        items.append(Item("Failsafe / radio loss (ST fault, stop, block)", "no radio since boot (block=NO_RADIO)", "bad", st_src, authority.age_s))
    elif s.stop not in ("NONE",):
        items.append(Item("Failsafe / radio loss (ST fault, stop, block)",
                          f"no fault latched · last stop cause: {s.stop} (history, not a current fault)", "ok", st_src, authority.age_s))
    else:
        items.append(Item("Failsafe / radio loss (ST fault, stop, block)", "no radio loss or fault reported", "ok", st_src, authority.age_s))

    # Switches as the firmware reads them (CH).
    for label, key in (("ARM switch as read (CH arm)", "arm"), ("MODE switch as read (CH mode)", "mode")):
        if channels is None:
            items.append(not_available(label, ch_why, ch_src))
        else:
            items.append(Item(label, channels.get(key, "?"), "off", ch_src, ch_age,
                              detail="channel:raw value:how the firmware decodes it"))

    for label in ("RSSI", "TX power", "Battery voltage", "Receiver temperature"):
        items.append(not_sent(label))
    return items


def known_build_text(firmware: str | None) -> tuple[str, str]:
    """(text, level) for a HELLO fw id."""
    if not firmware:
        return "no HELLO line", "off"
    if firmware == BASELINE_FIRMWARE:
        return f"{firmware} · known build: {KNOWN_FIRMWARE[firmware]}", "ok"
    if firmware in KNOWN_FIRMWARE:
        return f"{firmware} · known build: {KNOWN_FIRMWARE[firmware]}", "warn"
    return f"{firmware} · NOT a build this repository knows", "warn"


def firmware_items(inputs: ControlInputs, tracker: ControlTracker, now: float) -> list[Item]:
    monitor = inputs.monitor
    authority = monitor.authority(now)
    src = "MOCK controller replay" if inputs.mock else "controller"
    items: list[Item] = []
    firmware = monitor.firmware()
    if not firmware:
        items.append(not_available("Firmware id (HELLO fw)", "no HELLO line received. The controller sends it only at boot or "
                                   "when asked ('?'); this window never asks (it is listen-only)", src))
        items.append(not_available("Is it a build this repository knows?", "firmware not identified", src))
    else:
        text, level = known_build_text(firmware)
        hello_age = None if tracker.hello_at is None else now - tracker.hello_at
        items.append(Item("Firmware id (HELLO fw)", firmware, "ok", src + " HELLO line", hello_age,
                          detail="Identity as the controller states it; not a check of what is flashed."))
        items.append(Item("Is it a build this repository knows?", text.split(" · ", 1)[1] if " · " in text else text, level, "client FIRMWARE_IDS"))
        rest = ", ".join(f"{k}={v}" for k, v in monitor.hello.items() if k != "fw")
        items.append(Item("HELLO fields", rest or "none", "off", src + " HELLO line"))

    if monitor.status is None or monitor.status_at is None:
        items.append(not_available("Uptime (ST t=)", "no status line received", src))
        items.append(not_available("Time since the last status line", "no status line received", src))
    else:
        age = max(0.0, now - monitor.status_at)
        seconds = monitor.status.device_ms / 1000.0
        if authority.fresh:
            items.append(Item("Uptime (ST t=)", f"{seconds:.1f} s", "ok", src + " ST line", age))
        else:
            items.append(Item("Uptime (ST t=)", f"last reported {seconds:.1f} s · NOT current", "warn", src + " ST line", age))
        items.append(Item("Time since the last status line", f"{age:.1f} s", "ok" if authority.fresh else "bad", src + " ST line",
                          detail="The controller sends ST at least every 0.2 s; this window calls it STALE after 0.6 s."))

    n = tracker.restarts
    if n == 0:
        items.append(Item("Restarts", f"none seen since this window began watching ({monitor.lines} lines read)", "ok",
                          "uptime (ST t=) going backwards",
                          detail="'None seen' is not 'none happened': a restart before this window started, or while "
                                 "no telemetry source was connected, cannot be seen."))
    else:
        since = "" if tracker.last_restart_at is None else f" · last {now - tracker.last_restart_at:.0f} s ago"
        drop = ("" if tracker.last_restart_from_ms is None else
                f" (uptime fell from {tracker.last_restart_from_ms / 1000:.1f} s to {tracker.last_restart_to_ms / 1000:.1f} s)")
        items.append(Item("Restarts", f"controller restarted {n} time{'s' if n != 1 else ''} this session{since}{drop}",
                          "bad" if tracker.recent_restart(now) else "warn", "uptime (ST t=) going backwards",
                          detail="Restarts were the 20 Nov failure. Check the supply (playbook PWR-1) before relying on the robot."))
    if tracker.source_changes:
        items.append(Item("Telemetry source changes", f"{tracker.source_changes} (restart detection starts again on each new connection)",
                          "off", "hub"))

    unknown = monitor.unknown_lines
    wifi = " The Wi-Fi build's NET lines are counted here: this monitor does not parse them." if firmware == WIFI_FIRMWARE else ""
    items.append(Item("Unrecognised or malformed lines", f"{unknown} of {monitor.lines} lines", "warn" if unknown else "ok",
                      "controller monitor", detail="Includes malformed ST/CH lines." + wifi))
    return items


def power_item() -> Item:
    return Item("Battery / supply voltage", POWER_TEXT, "off", "none: the firmware has no voltage sensing", None, False,
                "Do not connect USB and the battery together until PWR-2 has an answer (playbook ground rules).")


def communication_items(inputs: ControlInputs, now: float) -> list[Item]:
    monitor = inputs.monitor
    authority = monitor.authority(now)
    items: list[Item] = []
    if inputs.link_label is None:
        items.append(not_available("Telemetry source", "no controller telemetry source connected: connect it in SETUP > Controller & IMU"))
    else:
        link = st.robot_link_item(authority, inputs.link_label, inputs.link_error)
        items.append(Item("Telemetry source", inputs.link_label, "off", "hub (listen-only)"))
        items.append(Item("Telemetry state", link.text, link.level, "hub / controller ST line", authority.age_s if authority.fresh else None, detail=link.detail))
    if inputs.link_error:
        items.append(Item("Telemetry source error", inputs.link_error, "bad", "hub"))
    if authority.fresh:
        s = authority.status
        if s.pc_age_ms < 0:
            items.append(Item("PC command age (ST pc_age)", "no PC command received since boot", "off", "controller ST line", authority.age_s))
        else:
            late = s.active == "PC" and s.pc_age_ms > PC_COMMAND_TIMEOUT_MS
            items.append(Item("PC command age (ST pc_age)", f"{s.pc_age_ms} ms" + (" · beyond the firmware's 250 ms stop" if late else ""),
                              "bad" if late else "ok", "controller ST line", authority.age_s,
                              detail="Firmware behaviour: 250 ms without an accepted PC command stops the motors and latches PC_STALE."))
    else:
        items.append(not_available("PC command age (ST pc_age)", "controller telemetry is not fresh"))
    if monitor.last_pc_command_at is None:
        items.append(Item("PC drive commands seen sent (client log)", "none seen", "off", "client log TX lines"))
    else:
        items.append(Item("PC drive commands seen sent (client log)", "a PC client has sent drive commands", "warn",
                          "client log TX lines", max(0.0, now - monitor.last_pc_command_at),
                          detail="Sent is not the same as applied: only the controller's ST line says who drives."))
    return items


# ---------------------------------------------------------------- transport inventory


@dataclass(frozen=True)
class TransportRow:
    route: str
    path: str
    state: str
    evidence: str                    # HARDWARE / MOCK / FIXTURE / SIM / PROPOSED / BLOCKED / REJECTED
    detail: str
    seen_now: str = ""


def transport_inventory(inputs: ControlInputs, now: float) -> list[TransportRow]:
    authority = inputs.monitor.authority(now)
    firmware = inputs.monitor.firmware()
    label = inputs.link_label or ""
    live = inputs.link_label is not None and authority.fresh and not inputs.mock
    radio_seen = ("controller reports link=1 (qualified radio link)" if live and authority.status.link else
                  "controller reports the radio link DOWN" if live else "not observed now")
    usb_seen = (f"telemetry listener: {label}" if label[:3].upper() == "COM" else "no USB listener")
    wifi_seen = "the connected firmware identifies as the Wi-Fi build" if firmware == WIFI_FIRMWARE else "not observed now"
    policy = inputs.policy.get("mode", "OFF")
    shadow_missing = inputs.live_shadow_missing
    shadow_note = ("software only; the AI never drives the motors. Live shadow: BLOCKED"
                   + (f" ({len(shadow_missing)} missing input(s): {'; '.join(shadow_missing)})" if shadow_missing else
                      " (missing inputs not available from the hub)" if shadow_missing is None else ""))
    mock_note = " MOCK session: nothing shown here is from hardware." if inputs.mock else ""
    return [
        TransportRow("RADIO", "RadioMaster -> ELRS -> robot", "Working on battery (reported); the competition default",
                     HARDWARE, "Reported working on battery on the night of 20 Nov. Not logged and not run as the written test (MD-Q)."
                     + mock_note, radio_seen if not inputs.mock else "MOCK telemetry, not hardware"),
        TransportRow("A", "Laptop -> USB -> ESP32-C3", "Demonstrated (normal path); tethered",
                     HARDWARE, "Wired PC driving and both handoff directions (HO-USB-1). Failure cases (HO-USB-2) have no recorded "
                     "hardware result. Needs PWR-2 before USB and battery are used together.", usb_seen),
        TransportRow("B", "Laptop -> RadioMaster Pocket -> ELRS -> robot", "PROPOSED; nothing implemented",
                     PROPOSED, "USB joystick mode is output-only, and USB serial cannot be a trainer input in released EdgeTX. "
                     "The Lua-script route is possible but never demonstrated; no script or sender exists.", "not applicable"),
        TransportRow("C", "Laptop -> a second ELRS transmitter -> robot", "Rejected for now", REJECTED,
                     "One transmitter per binding, and the Pocket would lose override and disarm authority.", "not applicable"),
        TransportRow("D", "Laptop -> Wi-Fi -> ESP32-C3", "Experimental; not the default", HARDWARE,
                     "Drove untethered on 20 Nov, then restarted and dropped on battery (12 PC_STALE latches, 56 status dropouts, "
                     "3 logged restarts in two hours, then a restart loop).", wifi_seen),
        TransportRow("AI SHADOW", "Policy on the laptop, display only", "Software only; motors never driven by the AI", f"{MOCK} / {FIXTURE} / {SIM}",
                     shadow_note, f"policy mode: {policy}"),
    ]


# ---------------------------------------------------------------- readiness


def control_readiness(inputs: ControlInputs, tracker: ControlTracker, now: float) -> Readiness:
    result = _readiness(inputs, tracker, now)
    if inputs.mock:
        headline = "MOCK · " + result.headline
        if result.level == READY:
            return Readiness(WARNING, headline, "MOCK telemetry is a replay, not the robot: it is never Ready. " + result.detail, result.fix)
        return Readiness(result.level, headline, result.detail, result.fix)
    return result


def _readiness(inputs: ControlInputs, tracker: ControlTracker, now: float) -> Readiness:
    if inputs.link_label is None:
        return Readiness(NOT_CONFIGURED, "No controller telemetry source is connected",
                         "Without it this screen cannot tell who is driving. The RadioMaster still drives.",
                         "Connect the controller in SETUP > Controller & IMU (listen-only).")
    if inputs.link_error:
        return Readiness(FAILED, "Controller telemetry source failed", inputs.link_error,
                         "Reconnect in SETUP > Controller & IMU.")
    if tracker.recent_restart(now):
        return Readiness(FAILED, f"Controller restarted {tracker.restarts} time{'s' if tracker.restarts != 1 else ''} this session",
                         "Its uptime went backwards. Restarts were the 20 Nov failure.",
                         "Check the supply under load (playbook PWR-1); do not rely on the robot until it stays up.")
    authority = inputs.monitor.authority(now)
    if authority.freshness == "UNKNOWN":
        return Readiness(WARNING, "Waiting for the first controller status line", "", "Check the cable and the COM port.")
    if authority.freshness == "STALE":
        return Readiness(WARNING, f"Controller telemetry is stale ({authority.age_s:.1f} s)",
                         "Who is driving is not confirmed.", "Check the USB cable and that the controller is powered.")
    s = authority.status
    if "RADIO_LOST" in (s.fault, s.block):
        return Readiness(FAILED, "Radio lost", f"fault={s.fault} block={s.block} stop={s.stop}",
                         "Switch the RadioMaster on, then ARM off and on, and check the receiver.")
    if s.fault != "NONE":
        return Readiness(FAILED, f"Controller fault latched: {s.fault}", f"block={s.block} stop={s.stop}",
                         "Switch ARM off, then on, on the RadioMaster.")
    if not s.link:
        return Readiness(FAILED, "No qualified radio link", f"block={s.block}", "Switch the RadioMaster on and check the receiver.")
    conflicts = find_conflicts(s, None if inputs.monitor.last_pc_command_at is None else max(0.0, now - inputs.monitor.last_pc_command_at))
    if conflicts:
        return Readiness(WARNING, "Two command sources may conflict", "; ".join(conflicts),
                         "Close the PC client, or set MODE to RADIO on the RadioMaster.")
    if tracker.restarts:
        return Readiness(WARNING, f"Controller restarted {tracker.restarts} time{'s' if tracker.restarts != 1 else ''} earlier in this session",
                         "", "Check the supply (playbook PWR-1).")
    channels, _, why = _ch_state(inputs, tracker, now)
    if channels is None:
        return Readiness(WARNING, "Partial telemetry: no current CH line", why + ". Link quality and frame age are unknown.",
                         "Wait for a CH line (it is sent at least once a second).")
    rc_age = _int(channels, "rc_age")
    if rc_age is None or rc_age < 0 or rc_age >= RC_FRAME_TIMEOUT_MS:
        return Readiness(WARNING, "Radio frames are not arriving normally", f"CH rc_age={channels.get('rc_age')}", "Check the receiver.")
    if s.requested not in ("RADIO", "PC") or s.block in ("ARM_ON_AT_BOOT", "ARM_SWITCH_INVALID", "MODE_SWITCH_INVALID"):
        fix = {"ARM_ON_AT_BOOT": "Switch ARM off, then on.", "ARM_SWITCH_INVALID": "Move the ARM switch to a clear position.",
               "MODE_SWITCH_INVALID": "Move the MODE switch to a clear position."}.get(s.block, "Set MODE to RADIO on the RadioMaster.")
        return Readiness(WARNING, f"Controller is not accepting a source ({s.block})", f"requested={s.requested}", fix)
    firmware = inputs.monitor.firmware()
    detail = f"requested={s.requested} source={s.active} armed={int(s.armed)}; firmware " + (firmware if firmware else "not identified (no HELLO)")
    if s.active == "NONE":
        return Readiness(READY, f"Radio link up; {s.block.lower().replace('_', ' ')}: no source drives until it clears", detail)
    return Readiness(READY, f"Radio link up; {s.active} has control", detail)


# ---------------------------------------------------------------- gates


@dataclass(frozen=True)
class Prerequisite:
    key: str
    label: str
    met: bool
    reason: str                       # why it is unmet, or what met it


def _records(path: Path | None) -> list[dict[str, Any]]:
    module = motion_check_module()
    return module.load_safety_records(path) if module else []


def _latest_record(firmware: str | None, path: Path | None) -> dict[str, Any] | None:
    candidates = [r for r in _records(path) if firmware and r.get("firmware") == firmware]
    return max(candidates, key=lambda r: r.get("finished_unix", 0.0)) if candidates else None


def check_state(record: dict[str, Any] | None, name: str) -> tuple[str, str]:
    """('PASS' | 'FAIL' | 'NOT RUN', evidence text) for one wheels-raised check in one safety record."""
    entry = (record or {}).get("checks", {}).get(name) if record else None
    if not entry:
        return "NOT RUN", "no record"
    controller, operator = bool(entry.get("controller_pass")), bool(entry.get("operator_confirmed"))
    evidence = str(entry.get("controller_evidence", ""))
    if controller and operator:
        return "PASS", evidence
    if controller:
        return "FAIL", f"controller evidence passed but the operator did not confirm: {evidence}"
    return "FAIL", evidence or "controller evidence failed"


def autonomous_prerequisites(*, firmware: str | None, now_unix: float, record_path: Path | None = None,
                             command_path_verified: bool | None = None, observation_verified: bool | None = None,
                             confirmations: dict[str, bool] | None = None, mock: bool = False,
                             observation_missing: list[str] | None = None,
                             floor_permission: Callable[..., tuple[bool, str]] | None = None) -> list[Prerequisite]:
    """The five prerequisites (a) to (e) of AUTONOMOUS mode, each with its reason. Pure; reads only the safety record."""
    command_path_verified = COMMAND_PATH_VERIFIED if command_path_verified is None else command_path_verified
    observation_verified = OBSERVATION_PIPELINE_VERIFIED if observation_verified is None else observation_verified
    module = motion_check_module()
    permit = floor_permission or (module.floor_permission if module else None)
    out: list[Prerequisite] = []

    out.append(Prerequisite("command_path", "(a) A verified laptop-to-robot command path exists", command_path_verified,
                            "verified" if command_path_verified else
                            "none verified: the USB tether is demonstrated for the normal path only (failure cases not run), Wi-Fi was "
                            "unstable on 20 Nov, and the relay through the Pocket is PROPOSED and unimplemented"))

    if mock:
        takeover = Prerequisite("takeover", "(b) Takeover procedure verified on hardware (motion_check 'takeover')", False,
                                "MOCK session: nothing here is hardware")
    elif not firmware:
        takeover = Prerequisite("takeover", "(b) Takeover procedure verified on hardware (motion_check 'takeover')", False,
                                "the connected firmware has not identified itself")
    else:
        record = _latest_record(firmware, record_path)
        state, evidence = check_state(record, "takeover")
        age = None if record is None else now_unix - float(record.get("finished_unix", 0.0))
        if record is None:
            takeover = Prerequisite("takeover", "(b) Takeover procedure verified on hardware (motion_check 'takeover')", False,
                                    f"no safety record for firmware {firmware}: the takeover check has not been run")
        elif age is None or age > SAFETY_MAX_AGE_S or age < -60.0:
            takeover = Prerequisite("takeover", "(b) Takeover procedure verified on hardware (motion_check 'takeover')", False,
                                    f"the record on {firmware} is {age / 3600.0:.1f} h old (limit 12 h)")
        else:
            takeover = Prerequisite("takeover", "(b) Takeover procedure verified on hardware (motion_check 'takeover')",
                                    state == "PASS", f"takeover check {state}: {evidence}")
    out.append(takeover)

    if mock:
        out.append(Prerequisite("safety_checks", "(c) All five wheels-raised safety checks passed on this firmware within 12 h", False,
                                "MOCK session: nothing here is hardware"))
    elif permit is None:
        out.append(Prerequisite("safety_checks", "(c) All five wheels-raised safety checks passed on this firmware within 12 h", False,
                                "scripts/motion_check.py could not be imported"))
    else:
        allowed, why = permit(firmware, now_unix, record_path)
        out.append(Prerequisite("safety_checks", "(c) All five wheels-raised safety checks passed on this firmware within 12 h",
                                bool(allowed), why))

    detail = "; ".join(observation_missing) if observation_missing else ""
    out.append(Prerequisite("observation", "(d) The observation pipeline is verified (real IMU, own pose, opponent)", observation_verified,
                            "verified" if observation_verified else
                            "not verified: live shadow is BLOCKED (no live observation adapter; the baseline firmware sends no IMU; "
                            "own pose needs a calibrated map; there is no live opponent tracker)" + (f". Missing now: {detail}" if detail else "")))

    confirmations = confirmations or {}
    missing = [text for key, text in AUTONOMOUS_CONFIRMATIONS.items() if not confirmations.get(key)]
    out.append(Prerequisite("confirmations", "(e) The operator has explicitly confirmed", not missing,
                            "confirmed" if not missing else "not confirmed: " + "; ".join(missing)))
    return out


def autonomous_gate(**kwargs: Any) -> tuple[bool, list[str]]:
    """(allowed, reasons). Allowed only when every prerequisite is met. In this repository's state it is closed."""
    unmet = [f"{p.label}: {p.reason}" for p in autonomous_prerequisites(**kwargs) if not p.met]
    return not unmet, unmet


def autonomous_enable_offered(allowed: bool) -> bool:
    """Whether the GUI may offer to enable AUTONOMOUS. Never in this build, whatever the gate says."""
    return bool(allowed and AUTONOMOUS_ENABLEABLE_IN_THIS_BUILD)


# ---------------------------------------------------------------- motion check: safety record, plan, launch


@dataclass(frozen=True)
class CheckRow:
    name: str
    state: str                       # PASS / FAIL / NOT RUN
    evidence: str
    age_text: str


def safety_rows(firmware: str | None, now_unix: float, record_path: Path | None = None) -> tuple[list[CheckRow], str]:
    """The five wheels-raised checks from the safety record for this firmware, and one line saying what it is."""
    names = _required_checks()
    module = motion_check_module()
    record = _latest_record(firmware, record_path)
    if record is None:
        why = (f"no wheels-raised safety checks are recorded for firmware {firmware}" if firmware
               else "the connected firmware has not identified itself, so no record can be matched to it")
        return [CheckRow(name, "NOT RUN", "no record for this firmware" if firmware else "firmware unknown", "") for name in names], why
    age = now_unix - float(record.get("finished_unix", 0.0))
    rows = []
    for name in names:
        state, evidence = check_state(record, name)
        rows.append(CheckRow(name, state, evidence, f"{age / 3600.0:.1f} h ago"))
    if module is not None:
        allowed, why = module.floor_permission(firmware, now_unix, record_path)
    else:
        allowed, why = False, "scripts/motion_check.py could not be imported"
    return rows, ("FLOOR RUN PERMITTED: " if allowed else "Floor runs refused: ") + why


def motion_plan_rows(plan: Any, reference: dict[str, Any] | None) -> list[tuple[str, ...]]:
    """Plan table (what would be sent) joined with the simulator's prediction when it is available."""
    steps = {int(s["index"]): s for s in (reference or {}).get("steps", [])}
    rows = []
    for step in plan.steps:
        sim = steps.get(step.index)
        rows.append((str(step.index), step.tactic, f"{step.duration_s:.2f}", f"{step.throttle}", f"{step.steer}",
                     f"{step.left} / {step.right}",
                     f"{sim['forward_m']:+.3f}" if sim else "-", f"{sim['heading_change_deg']:+.1f}" if sim else "-",
                     sim["motion_class"] if sim else "-"))
    return rows


def compute_plan_and_reference() -> tuple[Any, dict[str, Any]]:
    """build_plan + sim_reference, in process. No hardware, no file written. Slow: call it from a worker."""
    module = motion_check_module()
    if module is None:
        raise RuntimeError("scripts/motion_check.py could not be imported")
    plan = module.build_plan()
    return plan, module.sim_reference(plan)


PORT_PATTERN = re.compile(r"^(COM\d{1,3}|/dev/tty[A-Za-z0-9_.\-]+)$", re.IGNORECASE)
CHECKS_CONFIRMATIONS = {
    "wheels_raised": "Both wheels are off the ground and free to spin",
    "power_cutoff": "The power cut-off is in my reach",
    "radiomaster": "The RadioMaster is in my hand and switched on",
}
FLOOR_CONFIRMATIONS = {
    "floor_clear": "The robot is on a clear, flat floor with at least 0.5 m free all round",
    "power_cutoff": "The power cut-off is in my reach",
    "radiomaster": "The RadioMaster is in my hand and switched on",
}


def validate_port(text: str) -> str | None:
    """None when ``text`` is an acceptable port name, else the reason."""
    if not text or not text.strip():
        return "select the controller's COM port (handoff_bench_client.py --list shows them)"
    if not PORT_PATTERN.match(text.strip()):
        return f"{text.strip()!r} is not a COM port name like COM5"
    return None


def power_problem(power: str | None) -> str | None:
    if not power:
        return "select how the robot is powered (usb-only, battery-and-usb or battery-only)"
    if power not in _power_choices():
        return f"unknown power mode {power!r}"
    if power == "battery-only":
        return "battery-only is not possible over USB: the cable is plugged in (motion_check refuses it); use usb-only or battery-and-usb"
    return None


def port_held_by_hub(link_label: str | None, link_error: str | None, port: str) -> bool:
    """True when the hub's own listener has this port open (label like 'COM5 (listen-only)')."""
    if not link_label or link_error or not port:
        return False
    return link_label.split(" ", 1)[0].strip().upper() == port.strip().upper()


def checks_command(port: str, power: str, python: str | None = None, script: Path | None = None) -> list[str]:
    """``python scripts/motion_check.py checks --port X --power Y`` (the arguments `checks` requires)."""
    return [python or sys.executable, str(script or MOTION_CHECK_SCRIPT), "checks", "--port", port.strip(), "--power", power]


def floor_command(port: str, power: str, python: str | None = None, script: Path | None = None) -> list[str]:
    """``python scripts/motion_check.py run --port X --surface floor --power Y``; motion_check refuses it without the record."""
    return [python or sys.executable, str(script or MOTION_CHECK_SCRIPT), "run", "--port", port.strip(), "--surface", "floor", "--power", power]


def launch_blockers(kind: str, *, port: str, power: str | None, confirmations: dict[str, bool], link_label: str | None,
                    link_error: str | None, mock: bool, process_running: bool, firmware: str | None = None,
                    now_unix: float = 0.0, record_path: Path | None = None) -> list[str]:
    """Everything that keeps a console launch disabled. Empty means the operator may press the button."""
    reasons: list[str] = []
    if mock:
        reasons.append("MOCK session: a console launched from here would talk to real hardware, which a MOCK window must not do")
    if process_running:
        reasons.append("a supervised console started from this window is still running")
    problem = validate_port(port)
    if problem:
        reasons.append(problem)
    elif port_held_by_hub(link_label, link_error, port):
        reasons.append(f"this Command Centre is listening on {port.strip()}; only one program can hold a COM port. Disconnect it "
                       "in SETUP > Controller & IMU first (the window then shows no telemetry)")
    problem = power_problem(power)
    if problem:
        reasons.append(problem)
    wanted = FLOOR_CONFIRMATIONS if kind == "floor" else CHECKS_CONFIRMATIONS
    missing = [text for key, text in wanted.items() if not confirmations.get(key)]
    if missing:
        reasons.append("tick every confirmation: " + "; ".join(missing))
    if kind == "floor":
        module = motion_check_module()
        if module is None:
            reasons.append("scripts/motion_check.py could not be imported")
        else:
            allowed, why = module.floor_permission(firmware, now_unix, record_path)
            if not allowed:
                reasons.append("floor run refused (motion_check refuses it too): " + why)
    return reasons


# ---------------------------------------------------------------- everything together


@dataclass(frozen=True)
class AutonomousBox:
    allowed: bool
    enable_offered: bool
    prerequisites: tuple[Prerequisite, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ControlStatus:
    authority: AuthorityView
    handoff: HandoffState
    elrs: list[Item]
    firmware: list[Item]
    power: Item
    communication: list[Item]
    transports: list[TransportRow]
    autonomous: AutonomousBox
    readiness: Readiness
    mock: bool


def build_control_status(inputs: ControlInputs, tracker: ControlTracker, now: float, now_unix: float,
                         record_path: Path | None = None) -> ControlStatus:
    tracker.observe(inputs.monitor, now)
    prereqs = autonomous_prerequisites(firmware=inputs.monitor.firmware(), now_unix=now_unix, record_path=record_path,
                                       mock=inputs.mock, observation_missing=inputs.live_shadow_missing)
    unmet = tuple(f"{p.label}: {p.reason}" for p in prereqs if not p.met)
    allowed = not unmet
    return ControlStatus(
        authority=authority_view(inputs, now), handoff=handoff_state(inputs, now),
        elrs=elrs_items(inputs, tracker, now), firmware=firmware_items(inputs, tracker, now), power=power_item(),
        communication=communication_items(inputs, now), transports=transport_inventory(inputs, now),
        autonomous=AutonomousBox(allowed, autonomous_enable_offered(allowed), tuple(prereqs), unmet),
        readiness=control_readiness(inputs, tracker, now), mock=inputs.mock)
