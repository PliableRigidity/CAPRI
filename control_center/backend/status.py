"""What the Operate screen is allowed to say. Pure functions: no Qt, no clock, no hardware.

Every item has a level: "ok", "warn", "bad" or "off" (not available / not in use).
The control source comes only from fresh controller telemetry; a click in the UI never changes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from control_center.backend.controller_link import Authority

TRACKING_LOST_AFTER_S = 0.5


@dataclass(frozen=True)
class Item:
    text: str
    level: str = "off"
    detail: str = ""


@dataclass(frozen=True)
class ControlDisplay:
    source: Item            # the big one: RADIO / PC / NONE / UNKNOWN / STALE
    armed: Item
    requested: Item
    detail: str = ""


@dataclass(frozen=True)
class Snapshot:
    control: ControlDisplay
    robot_link: Item
    camera: Item
    map: Item
    own_bot: Item
    opponent: Item
    policy: Item
    imu: Item
    recording: Item
    camera_map_caption: str
    recording_active: bool = False
    recording_elapsed_s: float = 0.0
    mock: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def control_display(authority: Authority) -> ControlDisplay:
    if authority.freshness == "UNKNOWN":
        return ControlDisplay(Item("UNKNOWN", "warn"), Item("UNKNOWN", "warn"), Item("UNKNOWN", "off"),
                              "No controller telemetry received. The RadioMaster and the controller still decide who drives.")
    if authority.freshness == "STALE":
        was = authority.status
        return ControlDisplay(Item("STALE", "bad"), Item("UNKNOWN", "warn"), Item("STALE", "off"),
                              f"Last controller report {authority.age_s:.1f} s ago (it said {was.active}, "
                              f"{'armed' if was.armed else 'disarmed'}). That is no longer confirmed.")
    status = authority.status
    source_level = {"RADIO": "ok", "PC": "warn", "NONE": "off"}[status.active]
    detail = ""
    if status.active == "NONE":
        detail = f"Nobody is driving: {status.block}" + (f" (fault {status.fault})" if status.fault != "NONE" else "")
    elif not status.link:
        detail = "Controller reports the radio link down"
    return ControlDisplay(
        Item(status.active, source_level),
        Item("ARMED" if status.armed else "DISARMED", "warn" if status.armed else "ok"),
        Item(status.requested, "ok" if status.requested == status.active else "warn"),
        detail)


def robot_link_item(authority: Authority, source_label: str | None, error: str | None = None) -> Item:
    if source_label is None:
        return Item("Not connected", "off", "No controller telemetry source configured (see Setup)")
    if error:
        return Item("Link error", "bad", f"{source_label}: {error}")
    if authority.freshness == "UNKNOWN":
        return Item("Waiting for telemetry", "warn", source_label)
    if authority.freshness == "STALE":
        return Item(f"Telemetry STALE ({authority.age_s:.1f} s)", "bad", source_label)
    radio = "radio link OK" if authority.status.link else "RADIO LINK DOWN"
    return Item(f"Telemetry live · {radio}", "ok" if authority.status.link else "bad", source_label)


def camera_item(state: str, detail: str, fps: float, frame_age_s: float | None) -> Item:
    if state == "LIVE":
        if frame_age_s is not None and frame_age_s > 1.0:
            return Item(f"Camera frozen ({frame_age_s:.1f} s since last frame)", "bad", detail)
        return Item(f"Camera live · {fps:.0f} fps", "ok", detail)
    if state == "MOCK":
        return Item("MOCK camera (synthetic, not live)", "warn", detail)
    if state == "REPLAY":
        return Item("REPLAY of a video file (not live)", "warn", detail)
    if state == "OPENING":
        return Item("Opening camera", "warn", detail)
    if state == "LOST":
        return Item("Camera lost", "bad", detail)
    return Item("No camera", "bad", detail)


def map_item(level: str, text: str, detail: str) -> Item:
    return Item(text, {"ok": "ok", "weak": "warn", "invalid": "bad", "none": "off"}[level], detail)


def camera_map_caption(camera_state: str, map_text: str) -> str:
    """The one-line caption over the view, e.g. "Camera live · Map not calibrated"."""
    left = {"LIVE": "Camera live", "MOCK": "MOCK camera (synthetic)", "REPLAY": "REPLAY video (not live)",
            "OPENING": "Opening camera", "LOST": "Camera lost"}.get(camera_state, "No camera")
    return f"{left} · {map_text}"


def own_bot_item(camera_ok: bool, map_level: str, pose_ok: bool | None, reason: str, visible_ids: list[int],
                 used_ids: list[int], since_valid_s: float | None) -> Item:
    """``pose_ok`` is None when no frame has been processed yet."""
    if not camera_ok:
        return Item("No tracking (no camera)", "off")
    if pose_ok is None:
        return Item("Waiting for first frame", "off")
    if pose_ok:
        return Item("Tracking · markers " + " ".join(str(i) for i in used_ids), "ok")
    seen = "markers " + " ".join(str(i) for i in visible_ids) + " seen" if visible_ids else "no markers seen"
    if map_level in ("none", "invalid"):
        return Item(f"No position (map not calibrated) · {seen}", "off", reason)
    if since_valid_s is None:
        return Item(f"Not tracking · {seen}", "warn", reason)
    if since_valid_s >= TRACKING_LOST_AFTER_S:
        return Item(f"LOST for {since_valid_s:.1f} s · {seen}", "bad", reason)
    return Item(f"Not tracking · {seen}", "warn", reason)


def opponent_item(live_tracker_available: bool) -> Item:
    if not live_tracker_available:
        return Item("Not available live", "off",
                    "The unmarked-robot tracker has only been run on recorded footage; no live opponent tracking.")
    return Item("Tracker running", "ok")


def policy_item(mode: str, input_source: str | None, checkpoint: str | None, reason: str = "") -> Item:
    if mode == "SIM":
        return Item(f"SIM · {checkpoint or 'no checkpoint'}", "warn", "Policy is reading SIMULATOR observations. Output is display only.")
    if mode == "LIVE SHADOW":
        return Item(f"LIVE SHADOW · {checkpoint or 'no checkpoint'}", "warn",
                    f"Input: {input_source}. Proposed actions are display only and are never sent to the robot.")
    return Item("OFF", "off", reason or "No policy is running")


def imu_item(present: bool, rate_hz: float | None, link_configured: bool) -> Item:
    if present:
        return Item("IMU stream live" + (f" · {rate_hz:.0f} Hz" if rate_hz else ""), "ok")
    if not link_configured:
        return Item("No IMU stream (no controller link)", "off")
    return Item("No IMU stream", "off", "The controller is not sending IMU lines (the handoff bench sketch has no IMU).")


def recording_item(active: bool, elapsed_s: float, dropped: int, errors: int, last_folder: str | None) -> Item:
    if active:
        problems = []
        if dropped:
            problems.append(f"{dropped} dropped")
        if errors:
            problems.append(f"{errors} write error(s)")
        text = f"RECORDING {format_elapsed(elapsed_s)}" + (" · " + ", ".join(problems) if problems else "")
        return Item(text, "bad" if errors else "warn" if dropped else "ok")
    return Item("Not recording", "off", f"Last round: {last_folder}" if last_folder else "")


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
