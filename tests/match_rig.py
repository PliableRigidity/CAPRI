"""Shared fakes for tests/test_cc_ai_model.py and tests/test_cc_match.py.

Everything here is SIMULATED and in-process: a fake policy engine (no torch, no checkpoint), a fake tracking screen with scripted
results, a fake link service with a scripted gate, a fake controller monitor and a manual clock. No camera, serial port or socket is
opened, nothing is sent to anything, and none of it is hardware verification.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]

from control_center.app import load_config  # noqa: E402
from control_center.backend.command_mapping import LIMITS, POLICY_TACTICS, intended_command  # noqa: E402
from control_center.backend.control_modes import Authority, AuthorityState, CommandPlan, GateItem, Mode, tactic_of  # noqa: E402
from control_center.backend.hub import HubOptions, OperationsHub  # noqa: E402
from control_center.backend.link_service import RequestResult  # noqa: E402
from control_center.backend.policy_engine import LOOPS, ModelInfo, PolicyDecision, PolicyEngine  # noqa: E402
from control_center.backend.wifi_link import Endpoint, LinkState  # noqa: E402
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT  # noqa: E402


class Clock:
    """A manual monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


def process(seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def make_hub(tmp_path: Path, *, settings: bool = False, session_log: bool = False) -> OperationsHub:
    """A MOCK hub (synthetic camera, never started) with every writable folder inside ``tmp_path``."""
    options = HubOptions(mock=True, start_camera=False, rounds_root=tmp_path / "rounds",
                         settings_dir=(tmp_path / "settings") if settings else None,
                         session_log_dir=(tmp_path / "sessions") if session_log else None)
    return OperationsHub(load_config(ROOT / "config" / "control_center.yaml"), options)


def publish_frame(hub: OperationsHub, state: str | None = None) -> None:
    """Publish a camera frame into the hub's capture session exactly as its camera thread would."""
    capture = hub.capture
    with capture._new_frame:
        capture._latest = (1, time.monotonic(), np.zeros((8, 8, 3), np.uint8))
        capture._new_frame.notify_all()
    if state:
        capture.state = state
    elif capture.state == "OPENING":
        capture.state = "MOCK"


# ---------------------------------------------------------------- the policy engine

def make_info(path: Path | str, *, loaded: bool = False, compatible: bool = True, problems: tuple[str, ...] = ()) -> ModelInfo:
    return ModelInfo(str(path), "ab" * 32, 64426, "2.9.2", "2.9.2", "2.15.0+cpu", "2.15.0+cpu", 19, "Discrete(5)", POLICY_TACTICS,
                     "sim_debug", "experimental5_v1", "masked_zero_v1", True, loaded, 1.0 if loaded else None, 0.25 if loaded else None,
                     compatible and not problems, problems, ())


class FakeEngine(PolicyEngine):
    """The real PolicyEngine's loop registry and decide() signature, with no checkpoint: ``load`` waits on ``gate`` (if given)."""

    def __init__(self, path: Path | str = RECOMMENDED_CHECKPOINT, *, gate: threading.Event | None = None, fail: str = "",
                 compatible: bool = True, tactic: str = "advance") -> None:
        super().__init__(path)
        self.gate, self.fail, self.compatible, self.tactic = gate, fail, compatible, tactic
        self.load_calls = 0
        self.load_threads: list[threading.Thread] = []
        self.decide_calls = 0

    def info(self) -> ModelInfo:
        if self.fail or not self.compatible:
            return make_info(self.path, compatible=False, problems=("scripted incompatibility",) if not self.compatible else ())
        return make_info(self.path, loaded=self.loaded)

    def load(self) -> ModelInfo:
        self.load_calls += 1
        self.load_threads.append(threading.current_thread())
        if self.gate is not None:
            self.gate.wait(10.0)
        if self.fail:
            self.load_error = self.fail
            return make_info(self.path, compatible=False, problems=(self.fail,))
        self._model = object()
        self._info = make_info(self.path, loaded=True)
        return self._info

    def decide(self, result, now: float | None = None) -> PolicyDecision:
        self.decide_calls += 1
        now = result.built_at if now is None else float(now)
        common = dict(source=result.source, ages_s=result.ages_s(), decided_at=now, input_time=result.oldest_input_t(), model_sha256="ab" * 32)
        if result.values is None:
            return PolicyDecision(None, None, None, {}, valid_for_control=False, blocked_by=tuple(result.blocked_by), observation=None, **common)
        index = POLICY_TACTICS.index(self.tactic)
        return PolicyDecision(self.tactic, index, 0.9, {t: (0.9 if t == self.tactic else 0.025) for t in POLICY_TACTICS},
                              valid_for_control=bool(result.valid_for_control), blocked_by=tuple(result.blocked_by),
                              observation=tuple(float(v) for v in result.values), **common)


# ---------------------------------------------------------------- tracking and telemetry (scripted)

class FakeWorker:
    def __init__(self) -> None:
        self.resumed = 0

    def resume(self) -> None:
        self.resumed += 1

    def isRunning(self) -> bool:  # noqa: N802 - mirrors QThread
        return True

    def start(self) -> None:
        pass


class FakeTelemetry:
    """The IMU as the tracking screen's telemetry reports it. Off by default: the baseline firmware sends none."""
    origin = "LIVE"
    mounting = SimpleNamespace(verified=True)

    def __init__(self, clock) -> None:
        self.clock = clock
        self.flow = False

    def latest(self, now: float):
        return SimpleNamespace(gyro_rad_s=(0.0, 0.0, 0.0), accel_m_s2=(0.0, 0.0, 9.81), t=self.clock()) if self.flow else None

    def report(self, now: float) -> dict[str, Any]:
        if self.flow:
            return {"present_detail": "IMU lines arriving", "fresh_detail": "fresh", "mounting_detail": "verified"}
        return {"present_detail": "controller status is arriving but there is no IMU line: this firmware does not send IMU data",
                "fresh_detail": "no IMU sample has been received", "mounting_detail": "NOT verified"}


class FakeTracking:
    """Stands in for the TRACKING screen: ``_result`` (fresh at the clock's time while ``flow``), ``telemetry``, ``config``, ``worker``,
    ``clear_opponent``. The opponent selection is cleared by ``clear_opponent`` and stays cleared until ``set`` selects it again."""

    def __init__(self, clock) -> None:
        self.clock = clock
        self.flow = True
        self.own_state, self.opponent_state = "measured", "tracked"
        self.own_xy, self.opponent_xy = (0.5, 0.5), (0.8, 0.4)
        self.telemetry = FakeTelemetry(clock)
        self.config = SimpleNamespace(arena={"width_m": 1.15, "height_m": 1.15})
        self.worker = FakeWorker()
        self.cleared = 0
        self._epoch = 0
        self._closed = False
        self._frozen: SimpleNamespace | None = None

    def set(self, own: str | None = None, opponent: str | None = None) -> None:
        if own is not None:
            self.own_state = own
        if opponent is not None:
            self.opponent_state = opponent

    def _lost_after_s(self) -> float:
        return 1.0

    def _make(self, t: float) -> SimpleNamespace:
        own_ok = self.own_state in ("measured", "predicted")
        own = {"state": self.own_state, "x_m": self.own_xy[0] if own_ok else None, "y_m": self.own_xy[1] if own_ok else None,
               "heading_deg": 10.0 if own_ok else None, "age_s": 0.02, "reason": "" if own_ok else f"scripted {self.own_state}",
               "used_ids": [1, 2], "visible_ids": [1, 2]}
        tracked = self.opponent_state == "tracked"
        opponent = {"state": self.opponent_state, "x_m": self.opponent_xy[0] if tracked else None,
                    "y_m": self.opponent_xy[1] if tracked else None, "confidence": 0.8 if tracked else None,
                    "reason": "" if tracked else f"scripted {self.opponent_state}", "image_based": True}
        return SimpleNamespace(t=t, done_at=t, epoch=0, own=own, opponent=opponent, limits={"predict_hold_s": 0.5})

    @property
    def _result(self) -> SimpleNamespace | None:
        if self._frozen is not None:
            return self._frozen
        return self._make(self.clock()) if self.flow else None

    def freeze_at(self, t: float) -> None:
        """Keep serving one result stamped at ``t`` (to age it)."""
        self._frozen = self._make(t)

    def clear_opponent(self) -> None:
        self.cleared += 1
        self.opponent_state = "not_selected"


class FakeMonitor:
    """The listen-only controller monitor: the RadioMaster link is reported UP and fresh."""

    def __init__(self, up: bool = True) -> None:
        self.up = up

    def authority(self, now: float):
        return SimpleNamespace(fresh=True, freshness="FRESH", status=SimpleNamespace(link=self.up))


# ---------------------------------------------------------------- the link service (scripted gate)

class FakeService:
    """The LinkService surface the MATCH tab uses, applied synchronously to a scripted mode. ``gate_open`` scripts the ModeManager's gate:
    closed (the real state of this repository) refuses AUTONOMOUS with reasons. Nothing here is a robot."""

    def __init__(self, clock, *, gate_open: bool = False, simulated: bool = True) -> None:
        self.clock = clock
        self.gate_open = gate_open
        self.simulated = simulated
        self.mode = Mode.MANUAL
        self.calls: list[tuple] = []
        self.decisions: list[tuple[Any, dict]] = []
        self.plans: list[CommandPlan] = []
        self._last: RequestResult | None = None
        self.heartbeat = 0.1

    def now(self) -> float:
        return self.clock()

    def is_running(self) -> bool:
        return True

    def gate_items(self) -> tuple[GateItem, ...]:
        if self.gate_open:
            return tuple(GateItem(n, True, "ok", k) for n, k in (("real_session", "SESSION"), ("acceptance", "HARDWARE"), ("build", "SOFTWARE")))
        return (GateItem("real_session", False, "this is a mock / replay / simulated session: nothing here is physical verification", "SESSION"),
                GateItem("acceptance", False, "no acceptance record is loaded", "HARDWARE"),
                GateItem("build", False, "this build does not enable autonomous mode", "SOFTWARE"))

    def latest(self) -> SimpleNamespace:
        gate = self.gate_items()
        authority = AuthorityState(Authority.AI if self.mode is Mode.AUTONOMOUS else Authority.NONE, "scripted")
        mode = SimpleNamespace(mode=self.mode, gate_open=self.gate_open, gate=gate, authority=authority, neutral_hold=False,
                               needs_confirmation=self.mode is not Mode.AUTONOMOUS)
        health = SimpleNamespace(state=LinkState.CONNECTED, heartbeat_age_s=self.heartbeat, protocol_ok=True, protocol_note="",
                                 error=None, endpoint=Endpoint())
        transmitting = any(p.sent and (p.transmitted_throttle or p.transmitted_steer) for p in self.plans[-3:])
        return SimpleNamespace(t=self.clock(), unix=0.0, running=True, simulated=self.simulated, mock_session=self.simulated, health=health,
                               mode=mode, gate=gate, authority=authority, controller=None,
                               acceptance=SimpleNamespace(complete=self.gate_open, missing=() if self.gate_open else ("step 4: not tested",)),
                               plans=tuple(self.plans), events=(), ever_sent=False, transmitting=transmitting, last_command_age_s=None,
                               last_request=self._last, worker_error=None)

    def request_mode(self, mode: Mode, confirmation: Any = None) -> None:
        self.calls.append(("mode", mode, confirmation))
        if mode is Mode.AUTONOMOUS:
            ok = self.gate_open and self.mode is Mode.AI_SHADOW and confirmation is not None
            self._last = RequestResult("mode AUTONOMOUS", ok, "" if ok else "gate closed; scripted refusal", 0.0)
            if ok:
                self.mode = Mode.AUTONOMOUS
        else:
            self.mode = mode
            self._last = RequestResult(f"mode {mode.value}", True, "", 0.0)

    def return_to_manual(self) -> None:
        self.calls.append(("manual",))
        self.mode = Mode.MANUAL

    def submit_decision(self, decision: Any, *, policy_loaded=None, mock=None, observation_at=None) -> None:
        self.decisions.append((decision, {"policy_loaded": policy_loaded, "mock": mock, "observation_at": observation_at}))
        tactic = tactic_of(decision)
        if tactic is None or self.mode is Mode.MANUAL:
            return
        intended = intended_command(tactic, LIMITS)
        sent = self.mode is Mode.AUTONOMOUS
        wire = intended if sent else (0, 0)
        why = "sent" if sent else "AI_SHADOW: displayed and logged only, nothing is transmitted"
        self.plans.append(CommandPlan(tactic, intended[0], intended[1], wire[0], wire[1], sent, why, self.clock(), self.mode.value,
                                      "AI" if sent else "NONE"))

    def drop_to_shadow(self) -> None:
        """What the ModeManager does when the authority is lost: AUTONOMOUS falls to AI_SHADOW."""
        if self.mode is Mode.AUTONOMOUS:
            self.mode = Mode.AI_SHADOW


def forget_loops() -> None:
    LOOPS.stop()
