"""The three operator-facing modes and who may drive. Pure Python: no Qt, no threads, no clock of its own, no port.

    MANUAL        the RadioMaster over ELRS drives; this program sends nothing (default on every start)
    AI_SHADOW     the policy proposes, the proposal is displayed and logged; every transmitted value is 0/0
    AUTONOMOUS    the policy's tactic becomes a Wi-Fi drive command; entered ONLY from AI_SHADOW, with every gate open
                  and a deliberate operator confirmation (typed phrase + three ticks, valid 30 s, single use)

There is no separate "PC" mode: PC-originated commands exist only as the implementation of AUTONOMOUS (and of the
supervised motion check, which is a different program). This is consistent with ``control_status``: MANUAL and AUTONOMOUS
are the two operating modes, AI SHADOW is the non-actuating display, and AUTONOMOUS is not enableable without hardware
acceptance records that do not exist in this repository's state.

Safety model
------------
* Authority (``ModeManager.authority``) is decided locally from what the CONTROLLER reports (the ST line) and from the
  state of this program. Physical disarm always wins: armed=0, MODE not PC, radio link down, any latched fault, a changed
  token, a stale controller report, or a failed / lost Wi-Fi link takes authority away (NONE or RADIO) at once, drops
  the mode to MANUAL, sends one neutral (if a link still exists) and latches: AUTONOMOUS needs a NEW confirmation.
* Failure of the AI loop, the tracking, the observation pipeline, the policy or the acceptance record also takes
  autonomous authority away at once (neutral, mode AI_SHADOW, with the reason). It is never silently held.
* The manager does NOT depend on the Wi-Fi link to stop the robot. A software stop button (``emergency_stop``) is
  carried by the same Wi-Fi link: once Wi-Fi has failed it cannot reach the robot. What stops the robot without Wi-Fi is the
  firmware: it stops the motors 250 ms after the last PC command, and the RadioMaster's disarm / MODE / radio-off. The
  tests check that losing the link invalidates authority locally.
* Nothing here sends a non-zero command except through ``WifiLink.send_drive`` (``compute_command``), and only while the
  authority is AI. In MANUAL and AI_SHADOW every plan has transmitted values 0/0 and ``sent`` False.
  The one thing sent in AI_SHADOW is, ONLY if the operator switched ``set_neutral_hold(True)`` on, a stream of 0,0 commands:
  the firmware accepts PC commands only after a neutral, and the gate needs the controller to say it accepts them.

Driving it: call ``tick(now)`` about every 20 ms from one thread (a QThread loop). ``tick`` calls ``link.tick`` itself. The
AI side calls ``submit(...)`` with each decision. Everything takes ``now`` on the ``time.monotonic()`` clock; tests inject one.

Not thread-safe by design: call every method from the one thread that runs ``tick`` (marshal operator button presses to it
with a queue or a Qt signal) and hand ``snapshot()`` results to the GUI.

Build lock: besides the hardware acceptance record, the gate has a ``build`` item that follows
``control_status.AUTONOMOUS_ENABLEABLE_IN_THIS_BUILD`` (False in this repository's state), so the screens' wording
"AUTONOMOUS: DISABLED in this build" stays true. Pass ``build_allows_autonomous=True`` only in tests.

Software tests of this module are NOT physical verification.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from control_center.backend.command_mapping import POLICY_TACTICS, Limits, MapperState, intended_command, map_tactic
from control_center.backend.acceptance import Acceptance, Requirement, motion_check_module, sketch_sha256
from control_center.backend.ai_observation import IMU_ABSENT_REASON, IMU_MAX_AGE_MS, ObservationStatus, observation_status
from control_center.backend.wifi_link import (DEFAULT_MAX_COMMAND, ControllerReport, LinkHealth, LinkState, SendResult,
                                              WifiLink)

CONFIRM_PHRASE = "ENABLE AUTONOMOUS"
CONFIRM_TTL_S = 30.0
AI_HEARTBEAT_MAX_AGE_S = 0.5
TOKEN_STABLE_S = 1.0
STATUS_FRESH_S = 0.5
PLAN_LOG_SIZE = 200
PLAN_LOG_REPEAT_S = 1.0
NEUTRAL_HOLD_MIN_PERIOD_S = 0.02

EVIDENCE_NOTE = ("Software tests and replays are NOT physical verification. Only the hardware acceptance record "
                 "(evidence HARDWARE) opens autonomous mode.")


class Mode(Enum):
    MANUAL = "MANUAL"
    AI_SHADOW = "AI_SHADOW"
    AUTONOMOUS = "AUTONOMOUS"


class Authority(Enum):
    RADIO = "RADIO"
    AI = "AI"
    NONE = "NONE"


@dataclass(frozen=True)
class AuthorityState:
    authority: Authority
    reason: str


@dataclass(frozen=True)
class GateItem:
    name: str
    ok: bool
    reason: str            # plain language: why it is closed, or what satisfied it
    kind: str              # what the item rests on: HARDWARE (acceptance record) / LIVE (measured now) / SOFTWARE / SESSION


@dataclass(frozen=True)
class Confirmation:
    """The operator's deliberate act: the typed phrase and three ticks. Valid 30 s from ``created_at``, single use."""

    phrase: str
    wheels_clear: bool               # wheels raised, or the floor is clear all round
    radiomaster_in_hand: bool        # the RadioMaster is in hand and the ARM switch is reachable
    understands_stops: bool          # the AI stops when disarmed (and the robot stops by itself if Wi-Fi fails)
    created_at: float                # manager clock, set when the operator pressed the final button
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class AiFeed:
    """What the AI/tracking side tells the manager. Replace or update it as often as it changes."""

    heartbeat_at: float | None = None          # when the AI loop last completed an iteration
    decision: Any = None                       # stream_adapter.Decision, ShadowRunner dict, or Decision.to_trace()
    decided_at: float | None = None            # when that decision was made (same clock)
    observation_at: float | None = None        # optional: time of the OLDEST input the decision used
    observation: ObservationStatus | None = None   # optional: precomputed status (else computed from ``decision``)
    policy_loaded: bool = False
    mock: bool = False                         # the input is a replay / mock / simulator: the gate can never open


@dataclass(frozen=True)
class CommandPlan:
    tactic: str
    intended_throttle: int          # what the tactic table says at the command scale (before the cap)
    intended_steer: int
    transmitted_throttle: int       # what actually went on the wire for this plan (0 unless sent)
    transmitted_steer: int
    sent: bool
    why_not: str
    t: float = 0.0
    mode: str = ""
    authority: str = ""


@dataclass(frozen=True)
class GateInputs:
    now: float
    acceptance: Requirement | None
    link: LinkHealth | None
    controller: ControllerReport | None
    observation: ObservationStatus | None
    ai_heartbeat_age_s: float | None
    policy_loaded: bool
    mock_session: bool
    build_allows_autonomous: bool = False


@dataclass(frozen=True)
class ModeSnapshot:
    mode: Mode
    authority: AuthorityState
    gate_open: bool
    gate: tuple[GateItem, ...]
    needs_confirmation: bool
    neutral_hold: bool
    last_event: str
    recent_plans: tuple[CommandPlan, ...]
    evidence_note: str = EVIDENCE_NOTE


# ---------------------------------------------------------------- tactic -> command (motion_check's own mapping)

_TACTIC_CACHE: dict[tuple[str, float], tuple[int, int] | None] = {}


def tactic_command(tactic: str, scale: float | None = None) -> tuple[int, int] | None:
    """(throttle, steer) of 255 for a tactic at the command scale, via scripts/motion_check.build_plan.

    That is the simulator's own tactic table (``autobot.sim.env.tactic_to_drive``), the 40 % scaling and the firmware mixer
    check. None when it cannot be mapped (unknown tactic, or motion_check / the simulator cannot be imported): then nothing
    is sent.
    """
    module = motion_check_module()
    if module is None:
        return None
    scale = module.DEFAULT_SCALE if scale is None else scale
    key = (tactic, float(scale))
    if key not in _TACTIC_CACHE:
        try:
            sequence = "stop:0.6" if tactic == "stop" else f"stop:0.6,{tactic}:0.2,stop:0.6"
            plan = module.build_plan(sequence, scale)
            step = next(item for item in plan.steps if item.tactic == tactic)
            _TACTIC_CACHE[key] = (int(step.throttle), int(step.steer))
        except Exception:
            _TACTIC_CACHE[key] = None
    return _TACTIC_CACHE[key]


def tactic_of(decision: Any) -> str | None:
    """The proposed tactic name from a Decision, a ShadowRunner dictionary or a trace; None when there is none."""
    if decision is None:
        return None
    if isinstance(decision, str):
        return decision
    proposed = decision.get("proposed") if isinstance(decision, dict) else getattr(decision, "proposed", None)
    if isinstance(proposed, dict):
        name = proposed.get("tactic")
        return str(name) if name else None
    return None


# ---------------------------------------------------------------- the gate


def autonomous_gate(inputs: GateInputs) -> tuple[bool, list[GateItem]]:
    """(open, items). Open only when EVERY item is ok. Each closed item says why, in plain language."""
    items: list[GateItem] = []

    def add(name: str, ok: bool, reason: str, kind: str) -> None:
        items.append(GateItem(name, bool(ok), reason, kind))

    link, controller, obs = inputs.link, inputs.controller, inputs.observation
    mock = inputs.mock_session or (link is not None and not link.real)
    add("real_session", not mock,
        "live session with a real Wi-Fi link" if not mock else
        "this is a mock / replay / simulated session or a fake link: nothing here is physical verification", "SESSION")

    requirement = inputs.acceptance
    add("acceptance", requirement is not None and requirement.complete,
        "every acceptance step PASS from HARDWARE evidence" if requirement is not None and requirement.complete else
        ("no acceptance record is loaded" if requirement is None else requirement.text), "HARDWARE")

    if link is None:
        link_ok, link_reason = False, "there is no Wi-Fi link"
    elif link.state is not LinkState.CONNECTED:
        link_ok, link_reason = False, f"the Wi-Fi link is {link.state.value}" + (f": {link.error}" if link.error else "")
    elif link.heartbeat_age_s is None or link.heartbeat_age_s >= STATUS_FRESH_S:
        link_ok, link_reason = False, "no fresh ST heartbeat (older than 0.5 s)"
    elif link.protocol_ok is not True:
        link_ok, link_reason = False, f"firmware or protocol not confirmed: {link.protocol_note}"
    else:
        link_ok, link_reason = True, f"CONNECTED, heartbeat {1000 * link.heartbeat_age_s:.0f} ms old, firmware {link.firmware_id}"
    add("link", link_ok, link_reason, "LIVE")

    c = controller
    if c is None:
        ctrl_ok, ctrl_reason = False, "the controller has reported nothing (no ST line)"
    elif not c.fresh:
        ctrl_ok, ctrl_reason = False, f"the controller's report is {c.age_s:.1f} s old"
    elif not c.link_up:
        ctrl_ok, ctrl_reason = False, "the controller reports the radio link DOWN"
    elif c.fault != "NONE":
        ctrl_ok, ctrl_reason = False, f"the controller has a latched fault: {c.fault} (turn ARM off, then on, on the RadioMaster)"
    elif not c.armed:
        ctrl_ok, ctrl_reason = False, "the controller reports DISARMED"
    elif c.requested != "PC" or c.active != "PC" or not c.pc_ok:
        ctrl_ok, ctrl_reason = False, (f"the controller is not accepting PC commands (MODE asks {c.requested}, source {c.active}, "
                                       f"block {c.block})")
    elif c.token_stable_s is None or c.token_stable_s < TOKEN_STABLE_S:
        ctrl_ok, ctrl_reason = False, "the controller's token changed less than 1 s ago (a hand-over is still settling)"
    else:
        ctrl_ok, ctrl_reason = True, "radio up, armed, MODE=PC accepted, no fault, token stable"
    add("controller", ctrl_ok, ctrl_reason, "LIVE")

    takeover = requirement is not None and requirement.takeover_pass
    add("radio_override", takeover,
        "RADIO takeover test PASS from HARDWARE evidence" if takeover else
        "the RadioMaster takeover test has not PASSed on hardware for this firmware", "HARDWARE")

    disarm_ok = c is not None and c.fresh and requirement is not None and requirement.disarm_pass
    add("physical_disarm", disarm_ok,
        "the ARM switch is read from the controller now and the disarm test PASSed on hardware" if disarm_ok else
        ("the controller's ARM state is not being read right now" if c is None or not c.fresh else
         "the ARM-off test has not PASSed on hardware for this firmware"), "HARDWARE")

    age = inputs.ai_heartbeat_age_s
    add("ai_process", age is not None and age <= AI_HEARTBEAT_MAX_AGE_S,
        "the AI loop is alive" if age is not None and age <= AI_HEARTBEAT_MAX_AGE_S else
        ("the AI loop has never reported" if age is None else f"the AI loop's heartbeat is {age:.1f} s old (limit 0.5 s)"), "LIVE")

    add("tracking", obs is not None and obs.tracking_ok,
        obs.tracking_reason if obs is not None and obs.tracking_ok else
        (obs.tracking_reason if obs is not None else "no tracking result yet"), "LIVE")

    imu = None if obs is None else obs.groups.get("imu")
    imu_ok = imu is not None and imu.valid and imu.age_known and imu.age_ms is not None and imu.age_ms <= IMU_MAX_AGE_MS
    add("imu", imu_ok,
        f"real IMU data, {imu.age_ms:.0f} ms old" if imu_ok else
        (imu.reason if imu is not None and not imu.valid else
         "no IMU age is known" if imu is not None else IMU_ABSENT_REASON), "LIVE")

    obs_ok = obs is not None and obs.complete and not obs.stale and obs.fresh_known
    add("observation", obs_ok,
        "all 19 values present, every group fresh" if obs_ok else
        ("no observation yet" if obs is None else
         ("; ".join(obs.missing) if obs.missing else "; ".join(obs.notes) or "the observation is not confirmed fresh")), "LIVE")

    add("policy", inputs.policy_loaded, "policy checkpoint loaded" if inputs.policy_loaded else "no policy checkpoint is loaded",
        "SOFTWARE")
    add("build", inputs.build_allows_autonomous,
        "this build allows autonomous mode" if inputs.build_allows_autonomous else
        "this build does not enable autonomous mode (control_status.AUTONOMOUS_ENABLEABLE_IN_THIS_BUILD is False): "
        "it changes only together with a hardware result, not by a code edit alone", "SOFTWARE")
    return all(item.ok for item in items), items


# Items whose failure means the controller / link is no longer trusted: the mode falls to MANUAL. The rest (AI loop,
# tracking, observation, policy, acceptance) fall back to AI_SHADOW: the robot is not driven, the AI keeps displaying.
MANUAL_ITEMS = frozenset({"real_session", "link", "controller", "physical_disarm", "radio_override"})


def build_flag() -> bool:
    """The build-level lock shared with the ELRS & Control screen. False in this repository's state."""
    try:
        from control_center.backend import control_status
        return bool(control_status.AUTONOMOUS_ENABLEABLE_IN_THIS_BUILD)
    except Exception:
        return False


# ---------------------------------------------------------------- the manager


class ModeManager:
    def __init__(self, link: WifiLink | None, acceptance: Acceptance | None, *, clock: Callable[[], float] = time.monotonic,
                 unix_clock: Callable[[], float] = time.time, max_command: int | None = None,
                 tactic_mapper: Callable[[str], tuple[int, int] | None] | None = None,
                 controller_source: Callable[[float], ControllerReport | None] | None = None, mock_session: bool = False,
                 sketch_hash: str | None | Callable[[], str | None] = None, capabilities: Iterable[str] = (),
                 confirmation_ttl_s: float = CONFIRM_TTL_S, plan_log_size: int = PLAN_LOG_SIZE,
                 build_allows_autonomous: bool | None = None) -> None:
        self.link, self.acceptance = link, acceptance
        self._build_override = build_allows_autonomous      # None: follow control_status (False in this repository's state)
        self._clock, self._unix = clock, unix_clock
        self.max_command = (link.max_command if link is not None else DEFAULT_MAX_COMMAND) if max_command is None else int(max_command)
        self._custom_mapper = tactic_mapper             # None: the runtime mapping of control_center.backend.command_mapping
        self._mapper = tactic_mapper or self._default_intended
        self._map_state = MapperState()                  # slew limiting and neutral-first for the drive path
        self._controller_source = controller_source
        self.mock_session = mock_session
        self._sketch_hash = sketch_hash
        self._sketch_cache: str | None = None
        self._sketch_cached = False
        self.capabilities = tuple(capabilities)
        self.confirmation_ttl_s = confirmation_ttl_s
        self.mode = Mode.MANUAL
        self.feed = AiFeed()
        self.last_event = "started in MANUAL"
        self.events: deque[tuple[float, str]] = deque(maxlen=200)
        self._plans: deque[CommandPlan] = deque(maxlen=plan_log_size)
        self._last_logged: CommandPlan | None = None
        self._last_logged_at = float("-inf")
        self._used_nonces: set[str] = set()
        self._neutral_hold = False
        self._closed = False
        self._gate_cache: tuple[float, tuple[bool, list[GateItem]]] | None = None
        self._in_tick = False
        if link is not None:
            link.drive_allowed = self._drive_allowed

    # ------------------------------------------------------------ the runtime mapping (command_mapping.py)

    def _limits(self) -> Limits:
        cap = max(1, min(int(self.max_command), 128))
        return Limits(max_command=cap, operating_scale=cap / 255.0)

    def _default_intended(self, tactic: str) -> tuple[int, int] | None:
        """The tactic table value at the runtime scale (before slew / neutral-first). None for a tactic outside the policy's five."""
        if tactic not in POLICY_TACTICS:
            return None
        try:
            return intended_command(tactic, self._limits())
        except Exception:
            return None

    # ------------------------------------------------------------ small helpers

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def _event(self, now: float, text: str) -> None:
        self.last_event = text
        self.events.append((now, text))

    def _controller(self, now: float) -> ControllerReport | None:
        if self._controller_source is not None:
            return self._controller_source(now)
        return None if self.link is None else self.link.controller(now)

    def _sketch(self) -> str | None:
        if callable(self._sketch_hash):
            return self._sketch_hash()
        if self._sketch_hash is not None:
            return self._sketch_hash
        if not self._sketch_cached:
            self._sketch_cache, self._sketch_cached = sketch_sha256(), True
        return self._sketch_cache

    @property
    def plans(self) -> list[CommandPlan]:
        return list(self._plans)

    @property
    def neutral_hold(self) -> bool:
        return self._neutral_hold

    # ------------------------------------------------------------ AI side

    def submit(self, decision: Any, now: float | None = None, *, policy_loaded: bool | None = None,
               mock: bool | None = None, observation_at: float | None = None) -> None:
        """Called by the AI loop with each decision. Also counts as the AI loop's heartbeat."""
        now = self._now(now)
        self.feed.decision, self.feed.decided_at, self.feed.heartbeat_at = decision, now, now
        self.feed.observation = None
        self.feed.observation_at = observation_at
        if policy_loaded is not None:
            self.feed.policy_loaded = policy_loaded
        if mock is not None:
            self.feed.mock = mock

    def ai_heartbeat(self, now: float | None = None) -> None:
        self.feed.heartbeat_at = self._now(now)

    # ------------------------------------------------------------ the gate and authority

    def _observation(self, now: float) -> ObservationStatus | None:
        feed = self.feed
        if feed.observation is not None:
            return feed.observation
        if feed.decision is None:
            return None
        elapsed = 0.0 if feed.decided_at is None else max(0.0, now - feed.decided_at)
        return observation_status(feed.decision, elapsed_s=elapsed)

    def gate(self, now: float | None = None) -> tuple[bool, list[GateItem]]:
        now = self._now(now)
        if self._in_tick and self._gate_cache is not None and self._gate_cache[0] == now:
            return self._gate_cache[1]            # one evaluation per tick: the same inputs give the same answer
        link = self.link
        health = None if link is None else link.health(now)
        requirement = None
        if self.acceptance is not None:
            requirement = self.acceptance.required_for_autonomous(None if health is None else health.firmware_id,
                                                                  self._sketch(), self._unix(), self.capabilities)
        heartbeat = None if self.feed.heartbeat_at is None else max(0.0, now - self.feed.heartbeat_at)
        inputs = GateInputs(now, requirement, health, self._controller(now), self._observation(now), heartbeat,
                            self.feed.policy_loaded, self.mock_session or self.feed.mock,
                            build_flag() if self._build_override is None else bool(self._build_override))
        result = autonomous_gate(inputs)
        self._gate_cache = (now, result)
        return result

    def authority(self, now: float | None = None) -> AuthorityState:
        """RADIO, AI or NONE, from the controller's own report. Local: it does not need the Wi-Fi link to say NONE."""
        now = self._now(now)
        report = self._controller(now)
        if report is None:
            return AuthorityState(Authority.NONE, "no controller report: with no link to the robot nobody can say who is driving")
        if not report.fresh:
            return AuthorityState(Authority.NONE, f"the controller's report is {report.age_s:.1f} s old: not confirmed")
        if not report.armed:
            return AuthorityState(Authority.NONE, "the controller reports DISARMED: nobody drives")
        if not report.link_up:
            return AuthorityState(Authority.NONE, "the controller reports the radio link DOWN: nobody drives")
        if report.fault != "NONE":
            return AuthorityState(Authority.NONE, f"the controller has a latched fault ({report.fault}): nobody drives")
        if report.active == "RADIO":
            return AuthorityState(Authority.RADIO, "the controller reports the RadioMaster as the active source")
        if self.mode is not Mode.AUTONOMOUS:
            return AuthorityState(Authority.NONE, f"{self.mode.value}: the AI is not allowed to drive")
        link = self.link
        if link is None or link.state is not LinkState.CONNECTED:
            return AuthorityState(Authority.NONE, "the Wi-Fi link is not connected: the AI cannot reach the robot")
        if link.needs_rearm:
            return AuthorityState(Authority.NONE, f"the link needs a deliberate re-arm: {link.latched_reason}")
        if report.requested != "PC" or report.active != "PC" or not report.pc_ok:
            return AuthorityState(Authority.NONE, f"the controller is not accepting PC commands (MODE asks {report.requested}, "
                                                  f"source {report.active}, block {report.block})")
        open_, items = self.gate(now)
        if not open_:
            return AuthorityState(Authority.NONE, "a gate is closed: " + "; ".join(f"{i.name}: {i.reason}" for i in items if not i.ok))
        return AuthorityState(Authority.AI, "AUTONOMOUS: every gate is open and the controller accepts PC commands")

    def _drive_allowed(self) -> bool:
        """Installed on the link. Evaluated afresh at every send, so a stale approval cannot be used."""
        if self._closed or self.mode is not Mode.AUTONOMOUS:
            return False
        return self.authority(self._clock()).authority is Authority.AI

    # ------------------------------------------------------------ mode changes

    def _neutral(self, now: float) -> None:
        if self.link is not None:
            self.link.send_neutral(now, force=True)

    def _confirmation_problems(self, confirmation: Confirmation | None, now: float) -> list[str]:
        if confirmation is None:
            return ["a deliberate operator confirmation is required (type the phrase and tick all three boxes)"]
        problems = []
        if confirmation.phrase.strip() != CONFIRM_PHRASE:
            problems.append(f"the typed phrase must be exactly {CONFIRM_PHRASE!r}")
        for ok, text in ((confirmation.wheels_clear, "wheels raised, or the floor is clear all round"),
                         (confirmation.radiomaster_in_hand, "the RadioMaster is in your hand with the ARM switch reachable"),
                         (confirmation.understands_stops, "you understand the AI stops when disarmed")):
            if not ok:
                problems.append(f"tick: {text}")
        age = now - confirmation.created_at
        if age > self.confirmation_ttl_s:
            problems.append(f"the confirmation is {age:.0f} s old (valid for {self.confirmation_ttl_s:.0f} s): confirm again")
        elif age < -1.0:
            problems.append("the confirmation is dated in the future")
        if confirmation.nonce in self._used_nonces:
            problems.append("this confirmation was already used: confirmations are single use")
        return problems

    def request(self, mode: Mode, now: float | None = None, confirmation: Confirmation | None = None) -> tuple[bool, list[str]]:
        now = self._now(now)
        if self._closed:
            return False, ["the Command Centre is shutting down"]
        if not isinstance(mode, Mode):
            return False, [f"unknown mode {mode!r}"]
        if mode is self.mode:
            return True, []
        if mode is Mode.MANUAL:
            self._neutral(now)
            self._neutral_hold = False
            self._set_mode(Mode.MANUAL, now, f"{self.mode.value} -> MANUAL by the operator")
            return True, []
        if mode is Mode.AI_SHADOW:
            if self.mode is Mode.AUTONOMOUS:
                self._neutral(now)
            self._set_mode(Mode.AI_SHADOW, now, f"{self.mode.value} -> AI_SHADOW by the operator")
            return True, []
        # AUTONOMOUS
        if self.mode is not Mode.AI_SHADOW:
            return False, ["AUTONOMOUS cannot be entered directly from MANUAL: select AI_SHADOW first, then confirm"]
        open_, items = self.gate(now)
        reasons = [f"{item.name}: {item.reason}" for item in items if not item.ok]
        if not open_:
            return False, ["gate closed"] + reasons
        problems = self._confirmation_problems(confirmation, now)
        if problems:
            return False, problems
        ok, why = self.link.rearm(now, operator_confirmed=True) if self.link is not None else (False, "there is no Wi-Fi link")
        if not ok:
            return False, [why]
        self._used_nonces.add(confirmation.nonce)
        self._set_mode(Mode.AUTONOMOUS, now, "AI_SHADOW -> AUTONOMOUS: operator confirmed, every gate open")
        return True, []

    def _set_mode(self, mode: Mode, now: float, text: str) -> None:
        self.mode = mode
        self._gate_cache = None
        self._event(now, text)

    def _drop(self, target: Mode, reason: str, now: float) -> None:
        """Authority lost: one neutral at once, then the mode falls. Re-entering AUTONOMOUS needs a new confirmation."""
        self._neutral(now)
        if target is Mode.MANUAL:
            self._neutral_hold = False
        self._set_mode(target, now, f"AUTONOMOUS -> {target.value}: {reason}")

    def emergency_stop(self, now: float | None = None, reason: str = "operator pressed STOP") -> None:
        """Software stop. Note: it travels over the same Wi-Fi link; if Wi-Fi has failed it cannot reach the robot."""
        now = self._now(now)
        self._neutral(now)
        self._neutral_hold = False
        if self.mode is not Mode.MANUAL:
            self._set_mode(Mode.MANUAL, now, f"{self.mode.value} -> MANUAL: {reason}")
        else:
            self._event(now, f"MANUAL: {reason}")

    def set_neutral_hold(self, enabled: bool, now: float | None = None) -> tuple[bool, list[str]]:
        """AI_SHADOW only, operator's choice: stream 0,0 so the controller can accept PC commands. Never anything else."""
        now = self._now(now)
        if not enabled:
            self._neutral_hold = False
            return True, []
        problems = self._hold_problems(now)
        if problems:
            return False, problems
        self._neutral_hold = True
        self._event(now, "neutral hold ON: zero commands only")
        return True, []

    def _hold_problems(self, now: float) -> list[str]:
        if self.mode is not Mode.AI_SHADOW:
            return ["the neutral hold is only available in AI_SHADOW"]
        link, report = self.link, self._controller(now)
        problems = []
        if link is None or link.state is not LinkState.CONNECTED:
            problems.append("the Wi-Fi link is not CONNECTED")
        if report is None or not report.fresh:
            problems.append("the controller has no fresh report")
        else:
            if not report.armed or report.requested != "PC":
                problems.append("the RadioMaster must be armed with MODE on PC")
            if report.fault != "NONE" or not report.link_up:
                problems.append("the controller has a fault or no radio link")
        return problems

    # ------------------------------------------------------------ commands

    def _cap(self, throttle: int, steer: int) -> tuple[int, int]:
        """Scale both values together so the larger is at most max_command (the arc keeps its shape)."""
        biggest = max(abs(throttle), abs(steer))
        if biggest <= self.max_command or biggest == 0:
            return throttle, steer
        factor = self.max_command / biggest
        return int(round(throttle * factor)), int(round(steer * factor))

    def _observation_time(self, now: float) -> float | None:
        feed = self.feed
        if feed.observation_at is not None:
            return feed.observation_at
        if feed.decided_at is None:
            return None
        status = self._observation(now)
        age = None if status is None else status.oldest_input_age_ms()
        if age is None:
            return feed.decided_at
        return feed.decided_at - age / 1000.0

    def compute_command(self, decision: Any, now: float | None = None) -> CommandPlan:
        """Plan the command for a decision and, ONLY when the authority is AI, send it through the link's gated path."""
        now = self._now(now)
        authority = self.authority(now)
        tactic = tactic_of(decision)
        mapped = self._mapper(tactic) if tactic is not None else None
        intended = mapped if mapped is not None else (0, 0)
        label = tactic if tactic is not None else "none"

        def plan(sent: bool, why: str, wire: tuple[int, int] = (0, 0)) -> CommandPlan:
            built = CommandPlan(label, intended[0], intended[1], wire[0], wire[1], sent, why, now, self.mode.value,
                                authority.authority.value)
            self._log_plan(built, now)
            return built

        if tactic is None:
            return plan(False, "no tactic proposed: nothing to send")
        if mapped is None:
            return plan(False, f"the tactic {tactic!r} could not be mapped to a command: nothing sent")
        if self.mode is Mode.MANUAL:
            return plan(False, "MANUAL: the RadioMaster drives; the proposal is not transmitted")
        if self.mode is Mode.AI_SHADOW:
            return plan(False, "AI_SHADOW: displayed and logged only, nothing is transmitted")
        if authority.authority is not Authority.AI or self.link is None:
            return plan(False, f"authority is {authority.authority.value}: {authority.reason}")
        observation_time = self._observation_time(now)
        note = ""
        if self._custom_mapper is None:
            # The one runtime path: stale -> stop, slew-limited increases, neutral-first on a reversal, hard cap last.
            decided = observation_time if observation_time is not None else (self.feed.decided_at if self.feed.decided_at is not None else now)
            shaped = map_tactic(tactic, now, decided, self._limits(), self._map_state)
            wire, note = (shaped.throttle, shaped.steer), shaped.reason
        else:
            wire = self._cap(*mapped)
        result: SendResult = self.link.send_drive(wire[0], wire[1], now, observation_time)
        if result.sent:
            return plan(True, "sent" + (" (capped to max_command)" if wire != mapped and not note else "")
                        + (f" ({note})" if note and wire != mapped else ""), wire)
        return plan(False, f"the link refused it: {result.reason}")

    def _log_plan(self, plan: CommandPlan, now: float) -> None:
        last = self._last_logged
        same = last is not None and (last.tactic, last.transmitted_throttle, last.transmitted_steer, last.sent, last.why_not,
                                     last.mode) == (plan.tactic, plan.transmitted_throttle, plan.transmitted_steer, plan.sent,
                                                    plan.why_not, plan.mode)
        if same and now - self._last_logged_at < PLAN_LOG_REPEAT_S:
            return
        self._plans.append(plan)
        self._last_logged, self._last_logged_at = plan, now

    # ------------------------------------------------------------ the periodic work

    def _autonomous_violation(self, now: float) -> tuple[Mode, str] | None:
        report, link = self._controller(now), self.link
        if link is None:
            return Mode.MANUAL, "there is no Wi-Fi link"
        if report is None:
            return Mode.MANUAL, "the controller report is gone (the Wi-Fi link failed or was lost): authority is invalid"
        if not report.fresh:
            return Mode.MANUAL, f"the controller's report is {report.age_s:.1f} s old: authority is not confirmed"
        if not report.armed:
            return Mode.MANUAL, "physical disarm: the controller reports DISARMED"
        if report.active == "RADIO" or report.requested != "PC":
            return Mode.MANUAL, f"the RadioMaster took over: MODE asks {report.requested}, source {report.active}"
        if not report.link_up:
            return Mode.MANUAL, "radio loss: the controller reports the radio link DOWN"
        if report.fault != "NONE":
            return Mode.MANUAL, f"the controller latched fault {report.fault}"
        if link.state is not LinkState.CONNECTED:
            return Mode.MANUAL, f"the Wi-Fi link is {link.state.value}" + (f": {link.error}" if link.error else "")
        if link.needs_rearm:
            return Mode.MANUAL, f"the link latched: {link.latched_reason}"
        if not report.pc_ok:
            return Mode.MANUAL, f"the controller no longer accepts PC commands (block {report.block})"
        open_, items = self.gate(now)
        if not open_:
            failing = [i for i in items if not i.ok]
            target = Mode.MANUAL if any(i.name in MANUAL_ITEMS for i in failing) else Mode.AI_SHADOW
            return target, "; ".join(f"{i.name}: {i.reason}" for i in failing)
        return None

    def tick(self, now: float | None = None) -> None:
        """One 20 ms step: read the link, drop authority if anything failed, then plan and (if allowed) send."""
        now = self._now(now)
        if self._closed:
            return
        self._gate_cache = None
        if self.link is not None:
            self.link.tick(now)
        self._in_tick = True
        try:
            self._tick_body(now)
        finally:
            self._in_tick = False
            self._gate_cache = None

    def _tick_body(self, now: float) -> None:
        if self.mode is Mode.AUTONOMOUS:
            violation = self._autonomous_violation(now)
            if violation is not None:
                self._drop(violation[0], violation[1], now)
        if self._neutral_hold and (self.mode is not Mode.AI_SHADOW or self._hold_problems(now)):
            self._neutral_hold = False
            self._event(now, "neutral hold OFF: its conditions no longer hold")
        if self.mode is Mode.AUTONOMOUS:
            plan = self.compute_command(self.feed.decision, now)
            if not plan.sent and self.link is not None:
                self.link.send_neutral(now)         # keep the firmware's 250 ms timer fed with zeros while no command is sent
        elif self.mode is Mode.AI_SHADOW:
            self.compute_command(self.feed.decision, now)          # logged, never sent
            if self._neutral_hold and self.link is not None:
                self.link.send_neutral(now)

    def snapshot(self, now: float | None = None) -> ModeSnapshot:
        now = self._now(now)
        open_, items = self.gate(now)
        return ModeSnapshot(self.mode, self.authority(now), open_, tuple(items), self.mode is not Mode.AUTONOMOUS,
                            self._neutral_hold, self.last_event, tuple(self._plans)[-5:])

    def shutdown(self) -> None:
        """Application-exit hook. One neutral, mode MANUAL. Idempotent."""
        if self._closed:
            return
        now = self._clock()
        self._neutral(now)
        self._neutral_hold = False
        if self.mode is not Mode.MANUAL:
            self._set_mode(Mode.MANUAL, now, "shutdown: MANUAL")
        self._closed = True
