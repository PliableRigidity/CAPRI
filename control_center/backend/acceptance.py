"""The guided Wi-Fi acceptance workflow and its persisted record. Pure Python: no Qt, no port, no socket, nothing sent.

Autonomous mode stays BLOCKED until this record says, from HARDWARE evidence, that the Wi-Fi link was accepted. On
2026-11-20 the Wi-Fi build restarted, dropped out and the supply collapsed for a reason nobody has found
(artifacts/integration-2026-11-20/evidence.md), so nothing here may be satisfied by software.

Evidence kinds (``Evidence``):  HARDWARE (the real robot, a person present)  /  MOCK  /  SIM  /  OPERATOR-REPORTED.
  * ONLY ``record_result(..., evidence="HARDWARE")`` can store PASS (or FAIL: a real failure re-blocks later steps).
  * MOCK / SIM / OPERATOR-REPORTED evidence is stored, but it is stored as NOT TESTED with the note
    'simulated: not physical verification' (or the matching note) whatever status was asked for. It never changes the
    workflow, never replaces an existing HARDWARE result (it goes to the history), and can never produce PASS.
  * A step that needs the Wi-Fi link (2 to 6) counts only when the app observed a REAL link (``link_real=True``); a link
    built on an injected / fake transport is MOCK.

Steps, in order (a later step is BLOCKED while an earlier one has FAILED; a FAIL clears the results after it):
  1 baseline      RadioMaster-only baseline stable on battery. Needs the operator's confirmation and PWR-1 style readings
                  (pack at rest, converter output idle, converter output with the wheels running raised) and the minutes
                  of driving without a restart (the playbook's MD-Q asks for five).
  2 wifi          Wi-Fi established without motor commands (link CONNECTED, firmware identified, nothing but '?' sent).
  3 stability     heartbeats and stability over a measured window: the app measures it (``StabilityMeter``): at least 60 s
                  with an ST heartbeat younger than 0.5 s at every sample, state CONNECTED throughout, no controller
                  restart seen, on a REAL link.
  4 reception     command reception with motors disabled. BLOCKED: it needs a firmware feature (a motors-disabled mode that
                  the firmware reports) that no shipped sketch has. It stays BLOCKED until the caller says the connected
                  firmware reports it (``capabilities`` contains CAP_MOTORS_DISABLED_MODE). Nothing is invented.
  5 transitions   control-source transitions and failsafes; five sub-checks, each recorded on its own:
                  takeover (RADIO takeover), arm_off, radio_off, pc_link_loss, reconnect_rearm (a reconnect needs re-arm).
                  scripts/motion_check.py has the same five checks as wheels-raised records over USB or Wi-Fi
                  (stop, disarm, takeover, pc_link_loss, radio_loss). ``summary().motion_check`` says whether such a HARDWARE
                  record exists for the connected firmware and over which link; it is information only and is never
                  converted into a PASS here, because it may have been recorded over USB.
  6 supervised    supervised wheels-raised low-power movement. BLOCKED until 1 to 5 PASS. The supervised runner this step
                  expects (to be built; none exists yet) must: be a HOLD-TO-RUN dead-man (nothing is sent as soon as the key
                  is released or the window loses focus); never exceed 20 % of full command (51 of 255); move for at most 1 s
                  per move, then zero; send nothing non-zero unless the operator confirmed wheels raised, RadioMaster in
                  hand and ARM reachable; stop and stay stopped on any of: no status for 0.5 s, disarm, MODE away from PC,
                  latched fault, radio loss, changed token, outputs the move did not command; and write a log of every
                  command it sent. Its result is recorded here with evidence HARDWARE and the log path in ``data``.
  7 autonomous    DERIVED, never set: PASS only when steps 1 to 6 are PASS from fresh HARDWARE evidence for the connected
                  firmware. Otherwise BLOCKED, listing exactly which are missing.

Staleness: a result is stale (shown NOT TESTED: stale) when the firmware id differs from the one it was recorded on, when
the sketch hash differs (both known), and, for the stability and transition steps, after 12 hours. A result recorded
without a known firmware id can never satisfy ``required_for_autonomous``.

Persistence: JSON, one file for all profiles (default ``config/wifi_acceptance.json``, per profile key), written atomically.
A damaged file is ignored with a message and kept aside as ``<name>.damaged`` before the next save.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "config" / "wifi_acceptance.json"
WIFI_SKETCH = ROOT / "ELRS-Handoff-WiFi" / "ELRS-Handoff-WiFi.ino"
FILE_VERSION = 1

PASS, FAIL, BLOCKED, NOT_TESTED = "PASS", "FAIL", "BLOCKED", "NOT TESTED"
STATUSES = (PASS, FAIL, BLOCKED, NOT_TESTED)
STALE_AFTER_S = 12 * 3600.0
STABILITY_WINDOW_S = 60.0
STABILITY_MAX_HEARTBEAT_AGE_S = 0.5
STABILITY_MAX_SAMPLE_GAP_S = 1.0
BASELINE_MIN_MINUTES = 5.0
CAP_MOTORS_DISABLED_MODE = "motors_disabled_mode"


class Evidence:
    """The words for where a result came from. The UI shows these as they are."""

    HARDWARE = "HARDWARE"
    MOCK = "MOCK"
    SIM = "SIM"
    OPERATOR_REPORTED = "OPERATOR-REPORTED"
    ALL = (HARDWARE, MOCK, SIM, OPERATOR_REPORTED)
    TEXT = {
        HARDWARE: "HARDWARE: observed on the real robot with a person present",
        MOCK: "SIMULATED (mock): not physical verification",
        SIM: "SIMULATED (simulator): not physical verification",
        OPERATOR_REPORTED: "OPERATOR-REPORTED: said, not measured by this app; not physical verification",
    }
    NOTE_SIMULATED = "simulated: not physical verification"
    NOTE_OPERATOR_REPORTED = "operator-reported only: not measured by this app, not physical verification"

    @classmethod
    def describe(cls, kind: str) -> str:
        return cls.TEXT.get(kind, f"unknown evidence kind {kind!r}")


@dataclass(frozen=True)
class StepDef:
    number: int
    key: str
    title: str
    needs_real_link: bool
    subchecks: tuple[str, ...] = ()
    stale_after_s: float | None = None


SUBCHECKS = {
    "takeover": "RADIO takeover (MODE to RADIO stops the PC and the sticks drive)",
    "arm_off": "ARM off stops the motors at once",
    "radio_off": "Radio off stops the motors (failsafe)",
    "pc_link_loss": "PC link loss: the robot stops by itself within about 0.25 s",
    "reconnect_rearm": "A reconnect does not resume motion: it needs a deliberate re-arm",
}
STEPS: tuple[StepDef, ...] = (
    StepDef(1, "baseline", "RadioMaster-only baseline stable on battery", False),
    StepDef(2, "wifi", "Wi-Fi established without motor commands", True),
    StepDef(3, "stability", "Heartbeats and stability over a measured window", True, (), STALE_AFTER_S),
    StepDef(4, "reception", "Command reception with motors disabled", True),
    StepDef(5, "transitions", "Control-source transitions and failsafes", True, tuple(SUBCHECKS), STALE_AFTER_S),
    StepDef(6, "supervised", "Supervised wheels-raised low-power movement", True),
    StepDef(7, "autonomous", "Autonomous enable (derived)", True),
)
STEP_BY_KEY = {step.key: step for step in STEPS}
RECORDABLE = tuple(step.key for step in STEPS if step.key != "autonomous")
BASELINE_READINGS = ("pack_rest_v", "converter_out_idle_v", "converter_out_wheels_v", "minutes_without_restart")


def sketch_sha256(path: Path | None = None) -> str | None:
    """SHA-256 of the Wi-Fi sketch in this checkout. It says nothing about what is flashed on the board."""
    path = WIFI_SKETCH if path is None else path
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


@dataclass
class Result:
    key: str                              # "baseline" or "transitions.takeover"
    status: str
    evidence: str
    t_unix: float
    firmware_id: str | None = None
    sketch_sha256: str | None = None
    produced_by: str = ""
    notes: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    operator_confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "status": self.status, "evidence": self.evidence, "t_unix": self.t_unix,
                "firmware_id": self.firmware_id, "sketch_sha256": self.sketch_sha256, "produced_by": self.produced_by,
                "notes": self.notes, "data": self.data, "operator_confirmed": self.operator_confirmed}

    @classmethod
    def from_dict(cls, raw: Any) -> "Result | None":
        if not isinstance(raw, dict):
            return None
        try:
            status, evidence, key = str(raw["status"]), str(raw["evidence"]), str(raw["key"])
            if status not in STATUSES or evidence not in Evidence.ALL:
                return None
            result = cls(key, status, evidence, float(raw["t_unix"]), raw.get("firmware_id"), raw.get("sketch_sha256"),
                         str(raw.get("produced_by", "")), str(raw.get("notes", "")), dict(raw.get("data") or {}),
                         bool(raw.get("operator_confirmed", False)))
        except (KeyError, TypeError, ValueError):
            return None
        if result.status in (PASS, FAIL) and result.evidence != Evidence.HARDWARE:
            return None                                  # a hand-edited PASS/FAIL without hardware evidence is not believed
        return result


@dataclass(frozen=True)
class RecordOutcome:
    accepted: bool                        # the result was stored with the status that was asked for
    stored: Result | None
    status: str                           # the status that now stands for this check
    note: str


@dataclass(frozen=True)
class StepView:
    number: int
    key: str
    title: str
    status: str
    evidence: str | None
    t_unix: float | None
    firmware_id: str | None
    sketch_sha256: str | None
    produced_by: str
    notes: str
    reason: str                           # why BLOCKED / NOT TESTED / stale, or what stands
    subchecks: tuple["StepView", ...] = ()


@dataclass(frozen=True)
class MotionCheckInfo:
    record_exists: bool
    link: str | None
    age_h: float | None
    passed: tuple[str, ...]
    permitted: bool
    text: str


@dataclass(frozen=True)
class Requirement:
    complete: bool
    missing: tuple[str, ...]
    takeover_pass: bool
    disarm_pass: bool
    text: str


@dataclass(frozen=True)
class AcceptanceSummary:
    profile: str
    steps: tuple[StepView, ...]
    complete: bool
    missing: tuple[str, ...]
    load_message: str
    motion_check: MotionCheckInfo | None
    firmware_id: str | None
    evidence_note: str


# ---------------------------------------------------------------- the stability meter


class StabilityMeter:
    """Measures step 3 from LinkHealth samples. Feed it about once per second or faster.

    PASSED needs ``window_s`` of continuous samples with: state CONNECTED, an ST heartbeat younger than 0.5 s, no
    controller restart since the start, no sample gap above 1 s (a meter that was not fed measured nothing), and a REAL
    link at every sample. If the link was not real the run still shows progress but its evidence is MOCK.
    """

    IDLE, RUNNING, PASSED, FAILED = "IDLE", "RUNNING", "PASSED", "FAILED"

    def __init__(self, window_s: float = STABILITY_WINDOW_S) -> None:
        self.window_s = window_s
        self.reset()

    def reset(self) -> None:
        self.state = self.IDLE
        self.reason = "not started"
        self.started_at: float | None = None
        self.last_at: float | None = None
        self.samples = 0
        self.real_link = True
        self._restarts0 = 0
        self.max_heartbeat_age_s = 0.0

    @property
    def evidence(self) -> str:
        return Evidence.HARDWARE if self.real_link else Evidence.MOCK

    def progress_s(self, now: float) -> float:
        return 0.0 if self.started_at is None else max(0.0, (self.last_at if self.last_at is not None else now) - self.started_at)

    def start(self, health: Any, now: float) -> None:
        self.reset()
        self.state, self.reason = self.RUNNING, "measuring"
        self.started_at = self.last_at = now
        self._restarts0 = int(getattr(health, "restarts", 0) or 0)
        self.real_link = bool(getattr(health, "real", False))
        self._sample(health, now)

    def _fail(self, why: str) -> None:
        self.state, self.reason = self.FAILED, why

    def _sample(self, health: Any, now: float) -> None:
        state = getattr(getattr(health, "state", None), "value", getattr(health, "state", None))
        age = getattr(health, "heartbeat_age_s", None)
        self.samples += 1
        if not getattr(health, "real", False):
            self.real_link = False
        if state != "CONNECTED":
            self._fail(f"the link was {state}, not CONNECTED")
        elif age is None:
            self._fail("no heartbeat (ST line) was available")
        elif age >= STABILITY_MAX_HEARTBEAT_AGE_S:
            self._fail(f"the heartbeat was {age:.2f} s old (limit {STABILITY_MAX_HEARTBEAT_AGE_S} s)")
        elif int(getattr(health, "restarts", 0) or 0) > self._restarts0:
            self._fail("the controller restarted during the window")
        self.max_heartbeat_age_s = max(self.max_heartbeat_age_s, age or 0.0)

    def feed(self, health: Any, now: float) -> str:
        if self.state != self.RUNNING:
            return self.state
        if self.last_at is not None and now - self.last_at > STABILITY_MAX_SAMPLE_GAP_S:
            self._fail(f"no sample for {now - self.last_at:.1f} s: the window was not measured continuously")
            return self.state
        self.last_at = now
        self._sample(health, now)
        if self.state == self.RUNNING and now - (self.started_at or now) >= self.window_s:
            self.state = self.PASSED
            self.reason = (f"{self.window_s:.0f} s measured, {self.samples} samples, worst heartbeat "
                           f"{self.max_heartbeat_age_s:.2f} s, no restart")
        return self.state


# ---------------------------------------------------------------- the record


def motion_check_module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        import motion_check
    except Exception:
        return None
    return motion_check


class Acceptance:
    def __init__(self, path: Path | str | None = None, profile: str = "default", *,
                 unix_clock: Callable[[], float] = time.time, motion_check_path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.profile = profile
        self._unix = unix_clock
        self.motion_check_path = motion_check_path
        self.load_message = ""
        self._damaged = False
        self._results: dict[str, Result] = {}
        self._history: list[dict[str, Any]] = []
        self.load()

    # ------------------------------------------------------------ persistence

    def load(self) -> None:
        self._results, self._history, self.load_message, self._damaged = {}, [], "", False
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("profiles", {}), dict):
                raise ValueError("not an acceptance file")
            profile = raw.get("profiles", {}).get(self.profile, {})
            if not isinstance(profile, dict):
                raise ValueError("profile entry is not an object")
        except (OSError, ValueError) as error:
            self._damaged = True
            self.load_message = f"{self.path.name} is damaged and was ignored ({error}): every step shows NOT TESTED"
            return
        dropped = 0
        for key, entry in (profile.get("results") or {}).items():
            result = Result.from_dict(entry)
            if result is None or result.key != key or self._split(key) is None:
                dropped += 1
                continue
            self._results[key] = result
        history = profile.get("history")
        self._history = [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []
        if dropped:
            self.load_message = f"{dropped} unreadable or unbelievable record(s) in {self.path.name} were ignored"

    def _save(self) -> None:
        document: dict[str, Any] = {"version": FILE_VERSION, "profiles": {}}
        if self.path.is_file() and not self._damaged:
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("profiles"), dict):
                    document["profiles"] = existing["profiles"]
            except (OSError, ValueError):
                pass
        document["profiles"][self.profile] = {"results": {k: r.to_dict() for k, r in self._results.items()},
                                              "history": self._history[-500:]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._damaged and self.path.is_file():
            os.replace(self.path, self.path.with_name(self.path.name + ".damaged"))
            self._damaged = False
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(document, indent=1), encoding="utf-8")
        os.replace(temporary, self.path)

    # ------------------------------------------------------------ keys

    @staticmethod
    def _split(key: str) -> tuple[StepDef, str | None] | None:
        step_key, _, check = key.partition(".")
        step = STEP_BY_KEY.get(step_key)
        if step is None or step.key == "autonomous":
            return None
        if check:
            return (step, check) if check in step.subchecks else None
        return None if step.subchecks else (step, None)

    def _leaf_keys(self, step: StepDef) -> tuple[str, ...]:
        return tuple(f"{step.key}.{check}" for check in step.subchecks) if step.subchecks else (step.key,)

    # ------------------------------------------------------------ recording

    def record_result(self, step: str, status: str, evidence: str, *, check: str | None = None, produced_by: str = "operator",
                      firmware_id: str | None = None, sketch_sha256: str | None = None, notes: str = "",
                      data: dict[str, Any] | None = None, now_unix: float | None = None, link_real: bool | None = None,
                      operator_confirmed: bool = False, capabilities: Iterable[str] = ()) -> RecordOutcome:
        """Record one result. Only HARDWARE evidence can store PASS or FAIL. See the module docstring."""
        definition = STEP_BY_KEY.get(step)
        if definition is None or definition.key == "autonomous":
            raise ValueError(f"step {step!r} is not recordable (derived steps are computed); use one of {RECORDABLE}")
        if status not in (PASS, FAIL, NOT_TESTED):
            raise ValueError(f"status must be PASS, FAIL or NOT TESTED, not {status!r}")
        if evidence not in Evidence.ALL:
            raise ValueError(f"evidence must be one of {Evidence.ALL}, not {evidence!r}")
        if definition.subchecks:
            if check not in definition.subchecks:
                raise ValueError(f"step {step} needs check= one of {definition.subchecks}")
        elif check is not None:
            raise ValueError(f"step {step} has no sub-checks")
        key = f"{step}.{check}" if check else step
        now_unix = self._unix() if now_unix is None else now_unix
        data = dict(data or {})

        def build(final_status: str, final_evidence: str, note: str) -> Result:
            return Result(key, final_status, final_evidence, now_unix, firmware_id, sketch_sha256, produced_by, note, data,
                          operator_confirmed)

        if evidence != Evidence.HARDWARE:
            note = (Evidence.NOTE_OPERATOR_REPORTED if evidence == Evidence.OPERATOR_REPORTED else Evidence.NOTE_SIMULATED)
            stored = build(NOT_TESTED, evidence, note + (f" ({notes})" if notes else ""))
            self._history.append({"event": "non-hardware record, not counted", **stored.to_dict(), "asked_status": status})
            existing = self._results.get(key)
            if existing is None or existing.evidence != Evidence.HARDWARE:
                self._results[key] = stored
            self._save()
            return RecordOutcome(False, stored, NOT_TESTED, note)
        blocked = self._blocked_reason(definition, self._effective_statuses(firmware_id, sketch_sha256, now_unix),
                                       frozenset(capabilities))
        if status != NOT_TESTED:
            problem = blocked or self._hardware_problem(definition, status, link_real, produced_by, operator_confirmed, data)
            if problem:
                refusal = build(NOT_TESTED, evidence, f"not recorded as {status}: {problem}")
                self._history.append({"event": "refused", **refusal.to_dict(), "asked_status": status})
                self._save()
                return RecordOutcome(False, None, self._leaf_status(key, firmware_id, sketch_sha256, now_unix),
                                     f"not recorded as {status}: {problem}")
        stored = build(status, evidence, notes)
        self._results[key] = stored
        self._history.append({"event": "recorded", **stored.to_dict()})
        if status == FAIL:
            self._invalidate_after(definition, now_unix)
        self._save()
        return RecordOutcome(True, stored, status, "recorded")

    @staticmethod
    def _hardware_problem(definition: StepDef, status: str, link_real: bool | None, produced_by: str,
                          operator_confirmed: bool, data: dict[str, Any]) -> str:
        if not produced_by.strip():
            return "who or what produced the result must be stated"
        if definition.needs_real_link and link_real is not True:
            return "the app did not observe a REAL Wi-Fi link (a mock, replay or disconnected link is not hardware)"
        if status == PASS:
            if not operator_confirmed:
                return "the operator has not confirmed that this was done on the real robot"
            if definition.key == "baseline":
                absent = [name for name in BASELINE_READINGS if not isinstance(data.get(name), (int, float))
                          or isinstance(data.get(name), bool)]
                if absent:
                    return "the PWR-1 style readings are missing: " + ", ".join(absent)
                if float(data["minutes_without_restart"]) < BASELINE_MIN_MINUTES:
                    return f"under {BASELINE_MIN_MINUTES:.0f} minutes of driving without a restart"
        return ""

    def record_stability(self, meter: StabilityMeter, *, firmware_id: str | None, sketch_sha256: str | None = None,
                         operator_confirmed: bool = False, produced_by: str = "app: StabilityMeter",
                         now_unix: float | None = None, capabilities: Iterable[str] = ()) -> RecordOutcome:
        """Turn a finished meter into the step 3 result. A running meter records nothing."""
        if meter.state not in (StabilityMeter.PASSED, StabilityMeter.FAILED):
            return RecordOutcome(False, None, NOT_TESTED, "the stability window is still being measured")
        status = PASS if meter.state == StabilityMeter.PASSED else FAIL
        return self.record_result("stability", status, meter.evidence, produced_by=produced_by, firmware_id=firmware_id,
                                  sketch_sha256=sketch_sha256, notes=meter.reason,
                                  data={"window_s": meter.window_s, "samples": meter.samples,
                                        "max_heartbeat_age_s": round(meter.max_heartbeat_age_s, 3)},
                                  now_unix=now_unix, link_real=meter.real_link, operator_confirmed=operator_confirmed,
                                  capabilities=capabilities)

    def _invalidate_after(self, definition: StepDef, now_unix: float) -> None:
        for later in STEPS:
            if later.number <= definition.number:
                continue
            for key in self._leaf_keys(later):
                dropped = self._results.pop(key, None)
                if dropped is not None:
                    self._history.append({"event": f"invalidated by the failure of step {definition.number}",
                                          "t_unix": now_unix, **dropped.to_dict()})

    # ------------------------------------------------------------ evaluation

    def _stale_reason(self, definition: StepDef, result: Result, firmware_id: str | None, sketch: str | None,
                      now_unix: float) -> str:
        if firmware_id is not None and result.firmware_id != firmware_id:
            return f"stale: recorded on firmware {result.firmware_id}, now {firmware_id}"
        if sketch is not None and result.sketch_sha256 != sketch:
            return "stale: the sketch hash changed since this was recorded"
        if definition.stale_after_s is not None:
            age = now_unix - result.t_unix
            if age > definition.stale_after_s:
                return f"stale: recorded {age / 3600.0:.1f} h ago (limit {definition.stale_after_s / 3600.0:.0f} h)"
            if age < -60.0:
                return "stale: recorded in the future (clock changed)"
        return ""

    def _leaf_status(self, key: str, firmware_id: str | None, sketch: str | None, now_unix: float) -> str:
        return self._leaf(key, firmware_id, sketch, now_unix)[0]

    def _leaf(self, key: str, firmware_id: str | None, sketch: str | None, now_unix: float) -> tuple[str, str, Result | None]:
        result = self._results.get(key)
        split = self._split(key)
        if result is None or split is None:
            return NOT_TESTED, "no result recorded", None
        definition = split[0]
        if result.status == FAIL:
            return FAIL, result.notes or "failed on hardware", result
        if result.status == PASS:
            stale = self._stale_reason(definition, result, firmware_id, sketch, now_unix)
            if stale:
                return NOT_TESTED, stale, result
            return PASS, ("" if firmware_id is not None else "firmware not identified: cannot confirm this applies to the "
                          "connected firmware"), result
        return NOT_TESTED, result.notes or "not tested", result

    def _effective_statuses(self, firmware_id: str | None, sketch: str | None, now_unix: float) -> dict[str, str]:
        """Status of each recordable step 1..6 from its recorded results (no blocking applied)."""
        out: dict[str, str] = {}
        for definition in STEPS[:6]:
            leaves = [self._leaf(key, firmware_id, sketch, now_unix)[0] for key in self._leaf_keys(definition)]
            if any(s == FAIL for s in leaves):
                out[definition.key] = FAIL
            elif all(s == PASS for s in leaves):
                out[definition.key] = PASS
            else:
                out[definition.key] = NOT_TESTED
        return out

    @staticmethod
    def _blocked_reason(definition: StepDef, effective: dict[str, str], capabilities: frozenset[str]) -> str:
        for earlier in STEPS[:definition.number - 1]:
            if effective.get(earlier.key) == FAIL:
                return f"step {earlier.number} ({earlier.title}) failed: fix and pass it again first"
        if definition.key == "reception" and CAP_MOTORS_DISABLED_MODE not in capabilities:
            return ("no firmware reports a motors-disabled mode: the baseline and Wi-Fi sketches have none, so command "
                    "reception cannot be tested with the motors disabled. Nothing is invented.")
        if definition.key == "supervised":
            not_passed = [f"{s.number} {s.key}" for s in STEPS[:5] if effective.get(s.key) != PASS]
            if not_passed:
                return "steps 1 to 5 must PASS first (not passed: " + ", ".join(not_passed) + ")"
        return ""

    # ------------------------------------------------------------ reading

    def _view(self, definition: StepDef, firmware_id: str | None, sketch: str | None, now_unix: float,
              effective: dict[str, str], capabilities: frozenset[str]) -> StepView:
        blocked = self._blocked_reason(definition, effective, capabilities)
        if definition.subchecks:
            children = []
            for check, title in SUBCHECKS.items():
                status, reason, result = self._leaf(f"{definition.key}.{check}", firmware_id, sketch, now_unix)
                if blocked:
                    status, reason = BLOCKED, blocked
                children.append(StepView(definition.number, f"{definition.key}.{check}", title, status,
                                         None if result is None else result.evidence, None if result is None else result.t_unix,
                                         None if result is None else result.firmware_id,
                                         None if result is None else result.sketch_sha256,
                                         "" if result is None else result.produced_by, "" if result is None else result.notes,
                                         reason))
            overall = BLOCKED if blocked else effective[definition.key]
            why = blocked or ("all five sub-checks PASS" if overall == PASS else
                              "failed: " + "; ".join(c.title for c in children if c.status == FAIL) if overall == FAIL else
                              "not yet passed: " + "; ".join(c.title for c in children if c.status != PASS))
            return StepView(definition.number, definition.key, definition.title, overall, None, None, None, None, "", "", why,
                            tuple(children))
        status, reason, result = self._leaf(definition.key, firmware_id, sketch, now_unix)
        if blocked and status != FAIL:
            status, reason = BLOCKED, blocked
        return StepView(definition.number, definition.key, definition.title, status,
                        None if result is None else result.evidence, None if result is None else result.t_unix,
                        None if result is None else result.firmware_id, None if result is None else result.sketch_sha256,
                        "" if result is None else result.produced_by, "" if result is None else result.notes, reason)

    def _derived(self, views: list[StepView], firmware_id: str | None) -> tuple[bool, list[str]]:
        missing: list[str] = []
        for view in views:
            if view.status != PASS:
                missing.append(f"step {view.number} {view.title}: {view.status}" + (f" ({view.reason})" if view.reason else ""))
            for child in view.subchecks:
                if child.status != PASS and view.status != PASS:
                    missing.append(f"   {child.key}: {child.status}")
            if view.status == PASS and view.evidence is not None and view.evidence != Evidence.HARDWARE:
                missing.append(f"step {view.number}: evidence is {view.evidence}, not HARDWARE")
        if firmware_id is None:
            missing.insert(0, "the connected firmware is not identified: no result can be matched to it")
        return not missing, missing

    def summary(self, firmware_id: str | None = None, sketch_sha256: str | None = None, now_unix: float | None = None,
                capabilities: Iterable[str] = (), *, include_motion_check: bool = True) -> AcceptanceSummary:
        now_unix = self._unix() if now_unix is None else now_unix
        caps = frozenset(capabilities)
        effective = self._effective_statuses(firmware_id, sketch_sha256, now_unix)
        views = [self._view(step, firmware_id, sketch_sha256, now_unix, effective, caps) for step in STEPS[:6]]
        complete, missing = self._derived(views, firmware_id)
        if complete:
            final = StepView(7, "autonomous", STEPS[6].title, PASS, Evidence.HARDWARE, None, firmware_id, sketch_sha256, "derived", "",
                             "steps 1 to 6 PASS from HARDWARE evidence")
        else:
            final = StepView(7, "autonomous", STEPS[6].title, BLOCKED, None, None, None, None, "derived", "",
                             "missing: " + "; ".join(m.strip() for m in missing))
        return AcceptanceSummary(self.profile, tuple(views + [final]), complete, tuple(missing), self.load_message,
                                 self.motion_check_info(firmware_id, now_unix) if include_motion_check else None, firmware_id,
                                 "Only HARDWARE results count. " + Evidence.describe(Evidence.MOCK) + ".")

    def required_for_autonomous(self, firmware_id: str | None = None, sketch_sha256: str | None = None,
                                now_unix: float | None = None, capabilities: Iterable[str] = ()) -> Requirement:
        """Every step 1 to 6 PASS from fresh HARDWARE evidence for THIS firmware. Closed in this repository's state."""
        summary = self.summary(firmware_id, sketch_sha256, now_unix, capabilities, include_motion_check=False)
        steps = {view.key: view for view in summary.steps}
        transitions = {child.key: child for child in steps["transitions"].subchecks}
        text = ("acceptance record complete: every step PASS from HARDWARE evidence" if summary.complete
                else "acceptance record incomplete: " + "; ".join(m.strip() for m in summary.missing))
        return Requirement(summary.complete, summary.missing, transitions["transitions.takeover"].status == PASS,
                           transitions["transitions.arm_off"].status == PASS, text)

    def motion_check_info(self, firmware_id: str | None, now_unix: float | None = None) -> MotionCheckInfo | None:
        """Does scripts/motion_check.py's wheels-raised HARDWARE record exist for this firmware? Information only."""
        module = motion_check_module()
        now_unix = self._unix() if now_unix is None else now_unix
        if module is None:
            return MotionCheckInfo(False, None, None, (), False, "scripts/motion_check.py could not be imported")
        records = module.load_safety_records(self.motion_check_path)
        candidates = [r for r in records if firmware_id and r.get("firmware") == firmware_id]
        if not candidates:
            where = "none" if not records else f"{len(records)} record(s), none for {firmware_id or 'an unidentified firmware'}"
            return MotionCheckInfo(False, None, None, (), False,
                                   f"No motion_check wheels-raised HARDWARE record for this firmware ({where}). "
                                   "Information only: it is never converted into a PASS here.")
        record = max(candidates, key=lambda r: r.get("finished_unix", 0.0))
        age = (now_unix - float(record.get("finished_unix", 0.0))) / 3600.0
        passed = tuple(name for name, item in (record.get("checks") or {}).items()
                       if item.get("controller_pass") and item.get("operator_confirmed"))
        permitted, why = module.floor_permission(firmware_id, now_unix, self.motion_check_path)
        link = record.get("link")
        return MotionCheckInfo(True, link, age, passed, bool(permitted),
                               f"motion_check record over {link}, {age:.1f} h old, checks passed: {', '.join(passed) or 'none'}. "
                               f"{why}. Information only: it may have been recorded over USB and is never converted into a PASS.")
