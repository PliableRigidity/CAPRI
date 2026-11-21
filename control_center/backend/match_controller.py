"""Start Round / Stop Round and what must be true first. Pure logic on top of ``control_modes.ModeManager``.

Rules (the operator's, in order of importance):

* A round starts only when the operator presses Start Round (``operator_action=True``). Connecting the camera,
  selecting an opponent or opening a tab never starts one.
* MANUAL: always allowed (a timer and a log; nothing is transmitted by this application).
* AI SHADOW: needs a working camera and a loaded model. Everything else may be missing; the AI panel then shows
  which inputs are missing and no decision is valid. Nothing is ever transmitted.
* AUTONOMOUS: needs every prerequisite below AND the ModeManager's own gate AND the operator's confirmation. When any
  is missing the round is started in AI SHADOW instead (``downgraded``), with the reasons, never silently in motion.
* During a round a failed prerequisite (tracking, observation, inference, camera, link) stops autonomous command
  generation at once: the ModeManager sends neutral where the link allows it, the mode falls to AI SHADOW and a new
  deliberate confirmation is needed to re-enter AUTONOMOUS. Nothing here remembers a last action, so nothing can be
  reused after a failure.
* Stop Round: neutral, MANUAL, the opponent selection is cleared (a new opponent every round), the log is closed.

A software stop travels over Wi-Fi like any command; when Wi-Fi is down it cannot reach the robot. The firmware's
stale-command timeout and the RadioMaster are the real backstops, and the ModeManager already drops authority locally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from control_center.backend.control_modes import Confirmation, Mode


@dataclass(frozen=True)
class MatchInputs:
    camera_ok: bool = False
    camera_detail: str = "camera not delivering frames"
    arena_trusted: bool = False                 # calibration loaded AND aligned AND confirmed this run
    arena_detail: str = "arena calibration not confirmed this run"
    own_valid: bool = False                     # own robot MEASURED or PREDICTED within limits
    own_detail: str = "own robot not tracked"
    opponent_valid: bool = False                # opponent TRACKED (UNCERTAIN / LOST do not count)
    opponent_detail: str = "opponent not selected"
    observation_complete: bool = False          # every group of the 19 values real and fresh
    observation_detail: str = "observation incomplete"
    model_loaded: bool = False
    model_detail: str = "policy not loaded"
    mapping_validated: bool = False             # command mapping self-check passed (software)
    mapping_detail: str = "command mapping not validated"
    wifi_ready: bool = False                    # link connected, fresh, firmware known
    wifi_detail: str = "Wi-Fi not connected"
    authority_ok: bool = False                  # ModeManager gate open (includes hardware acceptance)
    authority_detail: str = "autonomous gate closed"
    safety_done: bool = False                   # physical safety checks recorded as HARDWARE evidence
    safety_detail: str = "physical safety checks not completed"
    manual_available: bool = False              # RadioMaster link up as the controller reports it
    manual_detail: str = "controller not reporting the radio link"
    inference_error: str = ""                   # non-empty: the last inference failed


@dataclass(frozen=True)
class Prereq:
    name: str
    ok: bool
    detail: str
    shadow: bool          # needed to start AI SHADOW
    autonomous: bool      # needed to start AUTONOMOUS


def prerequisites(i: MatchInputs) -> list[Prereq]:
    return [
        Prereq("Camera operational", i.camera_ok, i.camera_detail, True, True),
        Prereq("Policy loaded", i.model_loaded and not i.inference_error, i.inference_error or i.model_detail, True, True),
        Prereq("LIVE arena calibration loaded and aligned", i.arena_trusted, i.arena_detail, False, True),
        Prereq("Own robot tracking valid", i.own_valid, i.own_detail, False, True),
        Prereq("Opponent selected and tracked", i.opponent_valid, i.opponent_detail, False, True),
        Prereq("Observation complete and fresh", i.observation_complete, i.observation_detail, False, True),
        Prereq("Command mapping validated", i.mapping_validated, i.mapping_detail, False, True),
        Prereq("Wi-Fi transport verified", i.wifi_ready, i.wifi_detail, False, True),
        Prereq("Control authority established", i.authority_ok, i.authority_detail, False, True),
        Prereq("Physical safety checks completed", i.safety_done, i.safety_detail, False, True),
        Prereq("Manual recovery / disarm available", i.manual_available, i.manual_detail, False, True),
    ]


def missing(i: MatchInputs, mode: Mode) -> list[str]:
    key = "autonomous" if mode is Mode.AUTONOMOUS else "shadow"
    return [f"{p.name}: {p.detail}" for p in prerequisites(i) if getattr(p, key) and not p.ok]


@dataclass(frozen=True)
class StartResult:
    started: bool
    mode: Mode | None            # the mode the round actually runs in
    requested: Mode
    downgraded: bool
    reasons: tuple[str, ...]

    @property
    def message(self) -> str:
        if not self.started:
            return "Round NOT started: " + "; ".join(self.reasons)
        if self.downgraded:
            return f"Round started in {self.mode.value} (AUTONOMOUS was refused): " + "; ".join(self.reasons)
        return f"Round started in {self.mode.value}"


@dataclass(frozen=True)
class MatchSnapshot:
    running: bool
    mode: Mode
    requested: Mode | None
    downgraded: bool
    round_number: int
    elapsed_s: float
    last_stop_reason: str
    faults: tuple[str, ...]
    needs_reconfirm: bool
    prerequisites: tuple[Prereq, ...]
    message: str


class MatchController:
    def __init__(self, modes: Any, *, clock: Callable[[], float] = time.monotonic, session_log: Any = None,
                 on_round_end: Callable[[], None] | None = None, on_round_start: Callable[[int], None] | None = None,
                 duration_s: float | None = None) -> None:
        self.modes = modes
        self._clock = clock
        self.log = session_log
        self.on_round_end = on_round_end
        self.on_round_start = on_round_start
        self.duration_s = duration_s          # None: the timer only counts up (the competition's round length is not assumed here)
        self.running = False
        self.requested: Mode | None = None
        self.downgraded = False
        self.round_number = 0
        self.started_at: float | None = None
        self.stopped_elapsed = 0.0
        self.last_stop_reason = ""
        self.faults: list[str] = []
        self.needs_reconfirm = False
        self.message = "No round running. Nothing is transmitted."

    # ---- helpers
    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    @property
    def mode(self) -> Mode:
        return getattr(self.modes, "mode", Mode.MANUAL)

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        return (self._now(now) - self.started_at) if self.running else self.stopped_elapsed

    def _record(self, kind: str, event: str, **data: Any) -> None:
        if self.log is not None:
            self.log.log(kind, event, **data)

    # ---- start / stop
    def start_round(self, requested: Mode, inputs: MatchInputs, now: float | None = None, confirmation: Confirmation | None = None,
                    *, operator_action: bool = False) -> StartResult:
        """Begin a round. ``operator_action`` must be True: only the Start Round button passes it."""
        now = self._now(now)
        if not operator_action:
            return StartResult(False, None, requested, False, ("a round starts only when the operator presses Start Round",))
        if self.running:
            return StartResult(False, None, requested, False, ("a round is already running: Stop Round first",))
        reasons: list[str] = []
        actual = requested
        downgraded = False
        if requested is Mode.AUTONOMOUS:
            reasons = missing(inputs, Mode.AUTONOMOUS)
            if not reasons:
                self.modes.request(Mode.AI_SHADOW, now)
                ok, why = self.modes.request(Mode.AUTONOMOUS, now, confirmation)
                if not ok:
                    reasons = list(why)
            if reasons:
                actual, downgraded = Mode.AI_SHADOW, True
        if actual is Mode.AI_SHADOW:
            shadow_missing = missing(inputs, Mode.AI_SHADOW)
            if shadow_missing:
                result = StartResult(False, None, requested, False, tuple(shadow_missing + (reasons if downgraded else [])))
                self._record("round", "start refused", requested=requested.value, reasons=list(result.reasons))
                self.message = result.message
                return result
            if self.mode is not Mode.AI_SHADOW:
                self.modes.request(Mode.AI_SHADOW, now)
        elif actual is Mode.MANUAL:
            self.modes.request(Mode.MANUAL, now)
        self.running, self.requested, self.downgraded = True, requested, downgraded
        self.started_at, self.round_number = now, self.round_number + 1
        self.faults.clear()
        self.needs_reconfirm = False
        self.last_stop_reason = ""
        if self.log is not None:
            self.log.start_round(actual.value, requested=requested.value, downgraded=downgraded, reasons=list(reasons),
                                 prerequisites={p.name: p.ok for p in prerequisites(inputs)})
        if self.on_round_start is not None:
            self.on_round_start(self.round_number)
        result = StartResult(True, actual, requested, downgraded, tuple(reasons))
        self.message = result.message
        return result

    def stop_round(self, reason: str = "operator pressed Stop Round", now: float | None = None) -> None:
        now = self._now(now)
        if not self.running:
            self.modes.request(Mode.MANUAL, now)          # always leave the robot's command source neutral
            return
        self.stopped_elapsed = self.elapsed(now)
        self.running = False
        self.modes.request(Mode.MANUAL, now)              # neutral first, then MANUAL
        self.last_stop_reason = reason
        self.message = f"Round {self.round_number} stopped: {reason}. MANUAL. Nothing is transmitted."
        if self.log is not None:
            self.log.stop_round(reason, elapsed_s=round(self.stopped_elapsed, 2))
        if self.on_round_end is not None:
            self.on_round_end()                           # the opponent selection does not survive a round

    # ---- while running
    def set_mode(self, mode: Mode, inputs: MatchInputs, now: float | None = None, confirmation: Confirmation | None = None) -> tuple[bool, list[str]]:
        """Change mode during a round (MANUAL and AI SHADOW freely; AUTONOMOUS only through the ModeManager's gate)."""
        now = self._now(now)
        if not self.running:
            return False, ["no round is running"]
        if mode is Mode.AUTONOMOUS:
            reasons = missing(inputs, Mode.AUTONOMOUS)
            if reasons:
                return False, reasons
            if self.mode is Mode.MANUAL:
                self.modes.request(Mode.AI_SHADOW, now)
        ok, why = self.modes.request(mode, now, confirmation if mode is Mode.AUTONOMOUS else None)
        if ok and mode is Mode.AUTONOMOUS:
            self.needs_reconfirm = False
        self._record("mode", f"-> {mode.value}", ok=ok, reasons=list(why))
        return ok, list(why)

    def feed_decision(self, decision: Any, inputs: MatchInputs, now: float | None = None, *, observation_at: float | None = None,
                      mock: bool = False) -> bool:
        """The AI loop's output for this frame. Passed on only while a round runs, only when inference did not fail and
        the decision exists; anything else is dropped here so no old action can ever be reused."""
        now = self._now(now)
        if not self.running or decision is None or inputs.inference_error:
            return False
        self.modes.submit(decision, now, policy_loaded=inputs.model_loaded, mock=mock, observation_at=observation_at)
        return True

    def tick(self, inputs: MatchInputs, now: float | None = None) -> list[str]:
        """Called every ~100 ms during a round. Returns the faults found in this call."""
        now = self._now(now)
        if not self.running:
            return []
        found: list[str] = []
        if self.mode is Mode.AUTONOMOUS:
            found = [f"{p.name}: {p.detail}" for p in prerequisites(inputs) if p.autonomous and not p.ok]
            if found:
                reason = "; ".join(found[:3])
                self.modes.request(Mode.AI_SHADOW, now)            # neutral where the link allows, no last action kept
                self.needs_reconfirm = True
                self.faults.extend(found)
                self.message = f"AUTONOMOUS STOPPED: {reason}. Now AI SHADOW; re-enter AUTONOMOUS deliberately when fixed."
                if self.log is not None:
                    self.log.fault("autonomous stopped", reasons=found)
        elif self.mode is Mode.AI_SHADOW and inputs.inference_error:
            found = [f"inference error: {inputs.inference_error}"]
            self.faults.extend(found)
            if self.log is not None:
                self.log.fault("inference error", detail=inputs.inference_error)
        if self.duration_s is not None and self.elapsed(now) >= self.duration_s:
            self.stop_round(f"round time of {self.duration_s:g} s reached", now)
        return found

    def snapshot(self, inputs: MatchInputs, now: float | None = None) -> MatchSnapshot:
        now = self._now(now)
        return MatchSnapshot(self.running, self.mode, self.requested, self.downgraded, self.round_number, self.elapsed(now),
                             self.last_stop_reason, tuple(self.faults[-5:]), self.needs_reconfirm, tuple(prerequisites(inputs)), self.message)

    def shutdown(self) -> None:
        """Application exit: a running round is stopped (neutral, MANUAL), the log closed by its owner."""
        if self.running:
            self.stop_round("application closed during the round")
        else:
            self.modes.request(Mode.MANUAL)
