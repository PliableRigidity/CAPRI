"""The thread service behind the CONNECTIONS screen (and the Match tab): ONE Wi-Fi link, ONE mode manager, ONE acceptance
record, ONE worker thread.

    UI thread                          worker QThread (the ONLY thread that touches link / manager / acceptance)
    ---------                          -----------------------------------------------------------------------
    service.connect(...)   -> queue -> drain requests, then manager.tick(now) at 50 Hz (which ticks the link),
    service.request_mode() -> queue    feed the stability meter, log events, publish an immutable ServiceSnapshot at 10 Hz
    service.latest()  <- snapshot (read-only)         snapshot_changed(ServiceSnapshot) / event_added(str) signals

Safety rules kept here (the backend modules enforce the rest; nothing in this file can open a gate they close):

* Nothing is created at construction: no thread, no socket. ``start()`` starts the one worker. The real UDP transport is
  created only inside ``WifiLink.connect()`` / ``discover()``, i.e. only when the operator presses Connect / Discover in the
  running application. A ``transport_factory`` (tests) or ``mock_session=True`` (the in-process ``MockTransport``) means the
  link is NOT real: ``ServiceSnapshot.simulated`` is True, evidence recorded from it is MOCK, never HARDWARE.
* The service starts in MANUAL with the link DISCONNECTED. ``auto_connect`` in the profile store is honoured only if it is
  stored True (the default is False); a connection only ever sends ``?`` status requests, and every connect / reconnect leaves
  the link in ``needs_rearm``: nothing here re-arms. ``reconnect()`` returns to MANUAL first.
* An exception in the worker is caught, logged, forces neutral + MANUAL and the loop carries on; it never dies silently.
* ``shutdown()`` is idempotent: stop the thread, then (single-threaded) neutral only if a command was ever sent, close the
  link, mode MANUAL. It is also hooked to ``QCoreApplication.aboutToQuit`` as a safety net.
* What is persisted: only the endpoint (host, port) and when it last answered, in the profile store's ``wifi`` section.
  Never a connection state, a mode or an authority (the store refuses those keys).
* Nothing is invented: a value the firmware does not send is absent (``LinkHealth.unavailable`` says why).

Everything here is software. Tests with fake transports and clocks are NOT physical verification.
"""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal

from control_center.backend import wifi_link as wl
from control_center.backend.acceptance import (PASS, Acceptance, AcceptanceSummary, Evidence, StabilityMeter, sketch_sha256)
from control_center.backend.control_modes import (AuthorityState, CommandPlan, GateItem, Mode, ModeManager, ModeSnapshot)
from control_center.backend.wifi_link import (DiscoveryResult, Endpoint, LinkHealth, LinkState, WifiLink, validate_endpoint)

TICK_S = 0.02                      # the worker's period: 50 Hz
SNAPSHOT_PERIOD_S = 0.1            # snapshots: 10 Hz
SUMMARY_PERIOD_S = 1.0             # the acceptance summary is recomputed at most this often (it reads the motion-check file)
EVENT_LOG_SIZE = 200
SNAPSHOT_EVENTS = 40
DISCOVER_TIMEOUT_S = 2.0
TRANSMITTING_WINDOW_S = 1.5        # the plan log repeats an unchanged plan at least once a second
ERROR_REPEAT_S = 2.0
ERROR_CLEAR_S = 10.0                # a reported internal error leaves the snapshot after this long without a repeat
OPEN_STATES = (LinkState.CONNECTING, LinkState.CONNECTED, LinkState.DEGRADED)

MOCK_HELLO = ("HELLO fw=autobot-handoff-wifi-1 arm=CH5:HIGH pc=CH6:HIGH pc_limit=128 pc_timeout_ms=250 rc_timeout_ms=250 "
              "wifi_ap=AutoBot-Handoff wifi_ok=1 wifi_channel=6 udp_port=4210")


# ---------------------------------------------------------------- the in-process stand-in (mock sessions only)


class MockTransport:
    """A fake robot that lives inside this process: answers ``?`` with HELLO and sends ST lines. No network, no hardware.

    It reports the safe state (RadioMaster drives: MODE=RADIO, DISARMED), so the autonomous gate can never open on it.
    Everything read from it is SIMULATED.
    """

    probe = True

    def __init__(self, endpoint: Endpoint, clock: Callable[[], float]) -> None:
        self.label = f"MOCK Wi-Fi {endpoint.label} (in-process replay, no network)"
        self.error: str | None = None
        self.closed = False
        self.written: list[bytes] = []
        self._clock = clock
        self._t0 = clock()
        self._pending: list[str] = []
        self._last_st = float("-inf")

    def _status_line(self, now: float) -> str:
        values = dict(t=5000 + int(max(0.0, now - self._t0) * 1000), tok=1, link=1, armed=0, req="RADIO", act="RADIO", pc_ok=0,
                      block="DISARMED", fault="NONE", stop="NONE", L=0, R=0, pc_age=0)
        return "ST " + " ".join(f"{key}={values[key]}" for key in wl.client.STATUS_KEYS)

    def write(self, data: bytes) -> int:
        if self.closed:
            return 0
        self.written.append(bytes(data))
        if data.strip() == b"?":
            self._pending.append(MOCK_HELLO)
            self._pending.append(self._status_line(self._clock()))
            self._last_st = self._clock()
        return len(data)

    def read(self, _size: int = 4096) -> bytes:
        if self.closed:
            return b""
        now = self._clock()
        lines, self._pending = self._pending, []
        if now - self._last_st >= 0.2:
            self._last_st = now
            lines.append(self._status_line(now))
        return b"".join((line + "\n").encode("ascii") for line in lines)

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------- what the UI receives


@dataclass(frozen=True)
class StabilityState:
    state: str                   # IDLE / RUNNING / PASSED / FAILED (StabilityMeter's words)
    reason: str
    progress_s: float
    window_s: float
    evidence: str                # HARDWARE only on a real link, else MOCK
    samples: int
    recorded: str                # what the acceptance record said about the finished window ("" while none)


@dataclass(frozen=True)
class RequestResult:
    """The outcome of the newest operator request, in the backend's own words. Refusals are shown, never swallowed."""

    what: str
    ok: bool
    text: str
    t_unix: float


@dataclass(frozen=True)
class Preferences:
    host: str | None
    port: int | None
    preferred_ssid: str | None
    auto_connect: bool
    problem: str                 # "" when the store was read cleanly (or never saved)


@dataclass(frozen=True)
class ServiceSnapshot:
    """Everything the screens show, frozen. Built on the worker thread; the UI only reads it."""

    t: float                              # service clock when built
    unix: float
    running: bool                         # the worker thread was running
    simulated: bool                       # mock session or a fake transport: nothing here is from the robot
    mock_session: bool
    health: LinkHealth
    mode: ModeSnapshot
    gate: tuple[GateItem, ...]
    authority: AuthorityState
    controller: Any                       # wifi_link.ControllerReport | None: what the firmware's ST line says
    acceptance: AcceptanceSummary
    stability: StabilityState
    plans: tuple[CommandPlan, ...]
    events: tuple[str, ...]
    ever_sent: bool                       # a drive datagram (neutral included) was written this session
    transmitting: bool                    # a NON-ZERO command went out within the last 1.5 s
    last_command_age_s: float | None      # age of the newest planned command that was really sent (None: none)
    last_request: RequestResult | None
    discovery: DiscoveryResult | None
    discovering: bool
    endpoint: Endpoint
    preferred_ssid: str | None
    store_note: str
    worker_error: str | None


# ---------------------------------------------------------------- the worker thread


class _LoopThread(QThread):
    def __init__(self, service: "LinkService") -> None:
        super().__init__()
        self._service = service
        self.setObjectName("link-service-worker")

    def run(self) -> None:
        service = self._service
        service._worker_entered()
        try:
            next_at = time.monotonic()
            while not service._stop.is_set():
                service._iteration()
                next_at += TICK_S
                delay = next_at - time.monotonic()
                if delay < -0.5:
                    next_at = time.monotonic()               # fell behind: do not spin to catch up
                elif delay > 0:
                    service._stop.wait(delay)
        finally:
            service._worker_left()


@dataclass
class _Request:
    kind: str
    args: tuple = ()
    kwargs: dict | None = None


class LinkService(QObject):
    snapshot_changed = Signal(object)         # ServiceSnapshot, 10 Hz (and at once after a request)
    event_added = Signal(str)                 # one timestamped diagnostics line
    discovery_finished = Signal(object)       # wifi_link.DiscoveryResult
    request_finished = Signal(object)         # RequestResult

    def __init__(self, store=None, profile: str = "live", transport_factory: Callable[[Endpoint], Any] | None = None,
                 mock_session: bool = False, clock: Callable[[], float] = time.monotonic,
                 sketch_hash: str | None | Callable[[], str | None] = None, acceptance_path: Path | str | None = None, *,
                 unix_clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 capabilities: Iterable[str] = (), motion_check_path: Path | str | None = None, hook_app_quit: bool = True,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.store, self.profile = store, profile
        self.mock_session = bool(mock_session)
        self._clock, self._unix, self._sleep = clock, unix_clock, sleep
        self._temp_dir: Path | None = None
        self.capabilities = tuple(capabilities)
        if transport_factory is not None:
            self._factory: Callable[[Endpoint], Any] | None = transport_factory
        elif self.mock_session:
            self._factory = lambda endpoint: MockTransport(endpoint, self._clock)
        else:
            self._factory = None                                  # the built-in UdpPort, created only inside connect()
        if acceptance_path is None and self.mock_session:
            # A mock session remembers nothing and knows nothing of the real records: a private empty record.
            self._temp_dir = Path(tempfile.mkdtemp(prefix="autobots-mock-acceptance-"))
            acceptance_path = self._temp_dir / "wifi_acceptance.json"
            if motion_check_path is None:
                motion_check_path = self._temp_dir / "no-safety-record.json"
        self._sketch_hash = sketch_hash
        self._sketch_cache: str | None = None
        self._sketch_cached = False

        self.link = WifiLink(Endpoint(), transport_factory=self._factory, clock=clock, unix_clock=unix_clock)
        self.acceptance = Acceptance(acceptance_path, profile, unix_clock=unix_clock,
                                     motion_check_path=None if motion_check_path is None else Path(motion_check_path))
        self.manager = ModeManager(self.link, self.acceptance, clock=clock, unix_clock=unix_clock,
                                   mock_session=self.mock_session, sketch_hash=self._sketch_value,
                                   capabilities=self.capabilities)
        self.meter = StabilityMeter()

        self._lock = threading.RLock()
        self._lifecycle = threading.Lock()
        self._requests: "queue.SimpleQueue[_Request]" = queue.SimpleQueue()
        self._decision_lock = threading.Lock()
        self._pending_decision: tuple | None = None
        self._events: deque[str] = deque(maxlen=EVENT_LOG_SIZE)
        self._stop = threading.Event()
        self._thread: _LoopThread | None = None
        self._closing = False
        self._closed = False
        self._restored = False
        self._workers_active = 0
        self._workers_peak = 0
        self.iterations = 0
        self._discover_thread: threading.Thread | None = None
        self._discovering = False
        self._discovery: DiscoveryResult | None = None
        self._last_request: RequestResult | None = None
        self._worker_error: str | None = None
        self._worker_error_at = float("-inf")
        self._error_log: dict[str, float] = {}
        self._prefs = Preferences(None, None, None, False, "")
        self._last_state: LinkState = self.link.state
        self._last_error: tuple[Any, Any] = (None, None)
        self._latch_seen: str | None = None
        self._manager_seen: tuple | None = None
        self._last_logged_text = ("", float("-inf"))
        self._connected_logged = False
        self._stability_confirmed = False
        self._stability_recorded = False
        self._stability_note = ""
        self._summary: tuple[tuple, float, AcceptanceSummary] | None = None
        self._last_snapshot_at = float("-inf")
        self._dirty = False                          # something notable happened: publish at once, not at the next 10 Hz tick
        self._latest: ServiceSnapshot | None = None
        self._latest = self._build_snapshot(self._clock())        # before any thread exists: safe on the caller's thread
        if hook_app_quit:
            app = QCoreApplication.instance()
            if app is not None:
                try:
                    app.aboutToQuit.connect(self.shutdown)
                except Exception:
                    pass

    # ================================================================ small public helpers

    def now(self) -> float:
        """The service clock (use it as ``Confirmation.created_at``)."""
        return self._clock()

    def latest(self) -> ServiceSnapshot:
        return self._latest                                       # type: ignore[return-value]

    def event_log(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._events)

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.isRunning()

    @property
    def simulated(self) -> bool:
        return self.mock_session or not self.link.real

    @property
    def closed(self) -> bool:
        return self._closed

    def preferences(self) -> Preferences:
        return self._prefs

    def _sketch_value(self) -> str | None:
        source = self._sketch_hash
        if callable(source):
            return source()
        if source is not None:
            return source
        if not self._sketch_cached:
            self._sketch_cache, self._sketch_cached = sketch_sha256(), True
        return self._sketch_cache

    def _worker_entered(self) -> None:
        with self._lock:
            self._workers_active += 1
            self._workers_peak = max(self._workers_peak, self._workers_active)

    def _worker_left(self) -> None:
        with self._lock:
            self._workers_active -= 1

    # ================================================================ lifecycle

    def start(self) -> bool:
        """Start the ONE worker thread. False (and nothing happens) if it is already running or the service is shut down."""
        with self._lifecycle:
            if self._closed or self._closing:
                return False
            if self._thread is not None and self._thread.isRunning():
                return False
            if not self._restored:
                self.restore_preferences()
            self._stop.clear()
            self._thread = _LoopThread(self)
            self._thread.start()
            return True

    def shutdown(self, timeout_s: float = 2.0) -> None:
        """Idempotent. Stops the worker, then: one neutral only if a command was ever sent, link closed, mode MANUAL."""
        with self._lifecycle:
            if self._closed or self._closing:
                return
            self._closing = True
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.isRunning() and QThread.currentThread() is not thread:
            if not thread.wait(int(max(0.1, timeout_s) * 1000)):
                self._log("the worker did not stop within the timeout: shutting the link down anyway")
        now = self._clock()
        try:
            if self.link.drive_command_ever_sent:
                self.manager.shutdown()                            # one neutral, MANUAL
                self.link.shutdown(now)
            else:
                self.link.shutdown(now)                            # nothing was sent: nothing is sent now
                self.manager.shutdown()
        except Exception as error:
            self._log(f"shutdown: {type(error).__name__}: {error}")
        discover = self._discover_thread
        if discover is not None and discover.is_alive() and discover is not threading.current_thread():
            discover.join(1.0)
        self._closed = True
        try:
            self._latest = self._build_snapshot(self._clock())
        except Exception:
            pass
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    # ================================================================ preferences (endpoint only)

    def restore_preferences(self) -> Preferences:
        """Read the remembered endpoint from the store. Never the old connection state; never an auto-arm.

        ``auto_connect`` is honoured only when it is stored as True (default False) and only on the first call; connecting
        sends status requests only and leaves the link needing a deliberate re-arm.
        """
        first = not self._restored
        self._restored = True
        host = port = ssid = None
        auto = False
        problem = ""
        if self.store is not None and not self.mock_session:
            try:
                loaded = self.store.load(self.profile)
                problem = loaded.load_problem or ""
                section = loaded.sections.get("wifi", {})
                host, port = section.get("host"), section.get("port")
                ssid = section.get("preferred_ssid")
                auto = section.get("auto_connect") is True
            except Exception as error:                              # a bad profile name or an unexpected store failure
                problem = f"the saved Wi-Fi settings could not be read ({error}): defaults are used"
        endpoint = None
        if host:
            endpoint, why = validate_endpoint(host, port if port is not None else wl.DEFAULT_PORT)
            if endpoint is None:
                problem = "; ".join(filter(None, [problem, f"the saved endpoint was ignored: {why}"]))
        self._prefs = Preferences(host if endpoint else None, endpoint.port if endpoint else None, ssid if isinstance(ssid, str) else None,
                                  bool(auto), problem)
        if problem:
            self._log(f"settings: {problem}")
        if endpoint is not None:
            self._submit("configure", endpoint.host, endpoint.port, persist=False)
        if auto and first:
            self._log("auto-connect is on in the saved settings: connecting (status requests only; nothing is armed)")
            self._submit("connect")
        return self._prefs

    def _persist_endpoint(self, endpoint: Endpoint) -> str:
        """"" on success (or when nothing is stored in this session), else a plain note."""
        if self.store is None or self.mock_session:
            return "" if self.store is None else "MOCK session: the endpoint is not remembered"
        try:
            result = self.store.update_section(self.profile, "wifi", {"host": endpoint.host, "port": endpoint.port})
        except Exception as error:
            return f"the endpoint could not be remembered ({error})"
        return "" if result.ok else f"the endpoint could not be remembered ({result.error})"

    # ================================================================ requests (all non-blocking: they enqueue)

    def _submit(self, kind: str, *args: Any, **kwargs: Any) -> bool:
        if self._closed or self._closing:
            return False
        self._requests.put(_Request(kind, args, kwargs or None))
        return True

    def connect(self, host: str | None = None, port: int | str | None = None) -> None:
        """Open the link to the configured (or given) endpoint. Sends status requests only. The operator's act."""
        self._submit("connect", host, port)

    def disconnect(self) -> None:
        self._submit("disconnect")

    def reconnect(self) -> None:
        """Deliberate operator action: MANUAL, disconnect, connect again. The link then needs a re-arm: never resumes."""
        self._submit("reconnect")

    def discover(self) -> None:
        """Probe ONLY the configured endpoint with ``?`` in a short-lived thread; the result arrives by signal."""
        self._submit("discover")

    def configure_endpoint(self, host: object, port: object = wl.DEFAULT_PORT) -> tuple[bool, str]:
        """Validate now, apply on the worker, remember in the profile store. (ok, why); ``why`` may carry a note when ok."""
        endpoint, why = validate_endpoint(host, port)
        if endpoint is None:
            return False, why
        state = self._latest.health.state if self._latest is not None else LinkState.DISCONNECTED
        if state in OPEN_STATES:
            return False, "disconnect first: the address cannot change while the link is open"
        note = self._persist_endpoint(endpoint)
        self._submit("configure", endpoint.host, endpoint.port, persist=False)
        return True, note

    def request_mode(self, mode: Mode | str, confirmation: Any = None) -> None:
        """AI SHADOW / AUTONOMOUS (the Match tab's buttons). The outcome, including every refusal reason, arrives as a
        ``RequestResult`` (signal ``request_finished`` and ``ServiceSnapshot.last_request``)."""
        if isinstance(mode, str):
            try:
                mode = Mode[mode.strip().upper().replace(" ", "_")]
            except KeyError:
                pass
        self._submit("mode", mode, confirmation)

    def return_to_manual(self) -> None:
        """Always works: neutral (if the link is connected) and MANUAL. A software stop cannot reach a robot whose Wi-Fi is
        down: the firmware's 250 ms stale-command stop and the RadioMaster are the backstops."""
        self._submit("manual")

    def set_neutral_hold(self, enabled: bool) -> None:
        self._submit("hold", bool(enabled))

    def submit_decision(self, decision: Any, *, policy_loaded: bool | None = None, mock: bool | None = None,
                        observation_at: float | None = None) -> None:
        """The AI loop's entry (latest wins). Forwarded to ``ModeManager.submit`` with the time of THIS call."""
        if self._closed or self._closing:
            return
        with self._decision_lock:
            self._pending_decision = (decision, self._clock(), policy_loaded, mock, observation_at)

    def record_acceptance(self, step: str, status: str, evidence: str, *, check: str | None = None, notes: str = "",
                          data: dict[str, Any] | None = None, produced_by: str = "operator",
                          operator_confirmed: bool = False) -> None:
        """Record one acceptance result. The backend decides: only HARDWARE evidence on a REAL link can store PASS/FAIL, and a
        simulated service downgrades HARDWARE to MOCK. The backend's refusal text arrives as the ``RequestResult``."""
        self._submit("record", step, status, evidence, check=check, notes=notes, data=dict(data or {}), produced_by=produced_by,
                     operator_confirmed=bool(operator_confirmed))

    def record_wifi_established(self, operator_confirmed: bool = False) -> None:
        """Step 2 from what this app observed (CONNECTED, firmware identified, only '?' sent). Needs the operator's confirmation."""
        self._submit("record_wifi", bool(operator_confirmed))

    def start_stability_window(self, operator_confirmed: bool = False) -> None:
        """Step 3: measure 60 s of heartbeats. Counts as hardware only on a real link; the result is recorded by the service."""
        self._submit("stability_start", bool(operator_confirmed))

    def cancel_stability_window(self) -> None:
        self._submit("stability_cancel")

    # ================================================================ the worker

    def pump(self, now: float | None = None) -> None:
        """One worker iteration, synchronously. For tests only: refused while the worker thread is running."""
        if self.is_running():
            raise RuntimeError("pump() is for tests; the worker thread is running")
        self._iteration(now)

    def _iteration(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        self.iterations += 1
        handled = self._guard("requests", self._drain_requests, now) or 0
        self._guard("decision", self._apply_decision)
        if not self._closed:
            self._guard("manager.tick", self.manager.tick, now)
        self._guard("stability", self._feed_meter, now)
        self._guard("events", self._observe, now)
        if handled or self._dirty or now - self._last_snapshot_at >= SNAPSHOT_PERIOD_S - 1e-9:
            self._guard("snapshot", self._publish, now)

    def _guard(self, stage: str, function: Callable[..., Any], *args: Any) -> Any:
        try:
            return function(*args)
        except Exception as error:                                  # never let the worker die silently
            self._fail_safe(stage, error)
            return None

    def _fail_safe(self, stage: str, error: BaseException) -> None:
        text = f"{stage}: {type(error).__name__}: {error}"
        now = time.monotonic()
        if now - self._error_log.get(text, float("-inf")) >= ERROR_REPEAT_S:
            self._error_log[text] = now
            if len(self._error_log) > 50:
                self._error_log.clear()
            self._log(f"INTERNAL ERROR in the link service ({text}): neutral and MANUAL forced, the loop continues")
        self._worker_error = text
        self._worker_error_at = now
        self._dirty = True
        try:
            self.manager.emergency_stop(self._clock(), "internal error in the link service")
        except Exception:
            try:
                self.link.send_neutral(self._clock(), force=True)
            except Exception:
                pass
            try:
                self.manager.mode = Mode.MANUAL
            except Exception:
                pass

    def _drain_requests(self, now: float) -> int:
        count = 0
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return count
            count += 1
            try:
                self._handle(request, now)
            except Exception as error:
                self._fail_safe(f"request {request.kind}", error)
                self._result(request.kind, False, f"internal error: {type(error).__name__}: {error}")

    def _apply_decision(self) -> None:
        with self._decision_lock:
            pending, self._pending_decision = self._pending_decision, None
        if pending is not None:
            decision, at, policy_loaded, mock, observation_at = pending
            self.manager.submit(decision, at, policy_loaded=policy_loaded, mock=mock, observation_at=observation_at)

    # ------------------------------------------------------------ request handlers (worker thread only)

    def _handle(self, request: _Request, now: float) -> None:
        handler = getattr(self, f"_do_{request.kind}", None)
        if handler is None:
            self._result(request.kind, False, f"unknown request {request.kind!r}")
            return
        handler(now, *request.args, **(request.kwargs or {}))

    def _result(self, what: str, ok: bool, text: str) -> None:
        result = RequestResult(what, ok, text, self._unix())
        with self._lock:
            self._last_request = result
        self._log(f"{what}: {'ok' if ok else 'REFUSED'}" + (f": {text}" if text else ""))
        self._emit(self.request_finished, result)

    def _apply_endpoint(self, host: object, port: object, persist: bool) -> tuple[bool, str]:
        endpoint, why = validate_endpoint(host, port)
        if endpoint is None:
            return False, why
        ok, why = self.link.configure(endpoint)
        if not ok:
            return False, why
        note = self._persist_endpoint(endpoint) if persist else ""
        self._log(f"endpoint set to {endpoint.label}" + (f" ({note})" if note else ""))
        return True, note

    def _do_configure(self, now: float, host: str, port: int, persist: bool = False) -> None:
        ok, why = self._apply_endpoint(host, port, persist)
        if not ok:
            self._result("set endpoint", False, why)

    def _do_connect(self, now: float, host: object = None, port: object = None) -> None:
        if host is not None or port is not None:
            current = self.link.endpoint
            ok, why = self._apply_endpoint(host if host is not None else current.host, port if port is not None else current.port, True)
            if not ok:
                self._result("connect", False, why)
                return
        if self.link.state in OPEN_STATES:
            self._result("connect", True, f"the link is already {self.link.state.value}")
            return
        self._cancel_meter("a new connection was started")
        if self.link.connect(now):
            self._result("connect", True, f"connecting to {self.link.endpoint.label}: status requests only, nothing that could "
                                         "move the robot is sent")
        else:
            self._result("connect", False, (self.link.error or "the connection could not be opened")
                         + (f" What to try: {self.link.hint}" if self.link.hint else ""))

    def _go_manual(self, now: float) -> None:
        self.manager.request(Mode.MANUAL, now)                     # neutral first if connected; hold off
        self.manager.set_neutral_hold(False, now)

    def _do_disconnect(self, now: float) -> None:
        self._cancel_meter("the operator disconnected")
        self._go_manual(now)
        self.link.disconnect(now)
        self._result("disconnect", True, "disconnected; mode MANUAL; the RadioMaster drives")

    def _do_reconnect(self, now: float) -> None:
        self._cancel_meter("the operator reconnected")
        self._go_manual(now)
        self.link.disconnect(now)
        if self.link.connect(now):
            self._result("reconnect", True, "reconnecting in MANUAL: the link needs a deliberate re-arm, nothing resumes by itself")
        else:
            self._result("reconnect", False, (self.link.error or "could not reconnect")
                         + (f" What to try: {self.link.hint}" if self.link.hint else ""))

    def _do_discover(self, now: float) -> None:
        if self._discovering:
            self._result("discover", False, "a probe is already running")
            return
        if self.link.state in OPEN_STATES:
            self._result("discover", False, "the link is open: disconnect before probing (a second socket would take the status stream)")
            return
        endpoint = self.link.endpoint
        self._discovering = True
        thread = threading.Thread(target=self._discover_body, args=(endpoint,), name="link-discover", daemon=True)
        self._discover_thread = thread
        self._log(f"probing {endpoint.label} with status requests only (no scanning, no broadcast)")
        thread.start()

    def _discover_body(self, endpoint: Endpoint) -> None:
        try:
            probe = WifiLink(endpoint, transport_factory=self._factory, clock=self._clock, unix_clock=self._unix)
            result = probe.discover(DISCOVER_TIMEOUT_S, sleep=self._sleep)
        except Exception as error:
            result = DiscoveryResult(endpoint, False, None, None, None, "", {}, False, 0,
                                     f"the probe failed: {type(error).__name__}: {error}", None)
        with self._lock:
            self._discovery = result
            self._discovering = False
        if result.found:
            self._log(f"probe: {endpoint.label} answered, firmware {result.firmware_id}"
                      + ("" if result.protocol_ok else f" (protocol: {result.protocol_note})"))
        else:
            self._log(f"probe: no usable answer from {endpoint.label}: {result.error}")
            if result.hint:
                self._log(f"what to try: {result.hint}")
        self._emit(self.discovery_finished, result)

    def _do_mode(self, now: float, mode: Any, confirmation: Any) -> None:
        ok, reasons = self.manager.request(mode, now, confirmation)
        name = getattr(mode, "value", str(mode))
        self._result(f"mode {name}", ok, "; ".join(reasons))

    def _do_manual(self, now: float) -> None:
        self.manager.emergency_stop(now, "operator pressed Return to MANUAL")
        connected = self.link.connected
        self._result("Return to MANUAL", True, "MANUAL; neutral sent" if connected else
                     "MANUAL; there is no open link, so nothing could be sent: the RadioMaster and the firmware's 250 ms stop "
                     "protect the robot")

    def _do_hold(self, now: float, enabled: bool) -> None:
        ok, why = self.manager.set_neutral_hold(enabled, now)
        self._result("neutral hold " + ("ON" if enabled else "OFF"), ok, "; ".join(why))

    # ---- acceptance

    def _link_real(self, health: LinkHealth) -> bool:
        return bool(health.real and not self.mock_session and health.state is LinkState.CONNECTED)

    def _do_record(self, now: float, step: str, status: str, evidence: str, *, check: str | None, notes: str,
                   data: dict[str, Any], produced_by: str, operator_confirmed: bool) -> None:
        health = self.link.health(now)
        label = f"acceptance {step}" + (f".{check}" if check else "")
        if self.simulated and evidence == Evidence.HARDWARE:
            evidence = Evidence.MOCK                                  # a simulated service can never produce hardware evidence
            notes = ("recorded from a SIMULATED session: " + notes).strip()
        try:
            outcome = self.acceptance.record_result(step, status, evidence, check=check, produced_by=produced_by,
                                                    firmware_id=health.firmware_id, sketch_sha256=self._sketch_value(), notes=notes,
                                                    data=data, now_unix=self._unix(), link_real=self._link_real(health),
                                                    operator_confirmed=operator_confirmed, capabilities=self.capabilities)
        except ValueError as error:
            self._result(label, False, str(error))
            return
        self._summary = None
        self._result(label + f" as {status} ({evidence})", outcome.accepted, outcome.note)

    def _do_record_wifi(self, now: float, operator_confirmed: bool) -> None:
        health = self.link.health(now)
        problems = []
        if health.state is not LinkState.CONNECTED:
            problems.append(f"the link is {health.state.value}, not CONNECTED")
        if not health.firmware_id or health.protocol_ok is not True:
            problems.append("the firmware is not identified (no HELLO, or the protocol is not confirmed)")
        if self.link.drive_command_ever_sent:
            problems.append("a drive datagram was already sent in this session: step 2 means Wi-Fi WITHOUT motor commands; "
                            "disconnect and connect again")
        if problems:
            self._result("acceptance wifi", False, "not recorded: " + "; ".join(problems))
            return
        notes = (f"link CONNECTED to {health.endpoint.label}, firmware {health.firmware_id}, protocol ok, heartbeat "
                 f"{1000 * (health.heartbeat_age_s or 0.0):.0f} ms; only status requests were sent")
        self._do_record(now, "wifi", PASS, Evidence.HARDWARE, check=None, notes=notes, data={}, produced_by="app: WifiLink observation",
                        operator_confirmed=operator_confirmed)

    def _do_stability_start(self, now: float, operator_confirmed: bool) -> None:
        health = self.link.health(now)
        if self.meter.state == StabilityMeter.RUNNING:
            self._result("stability window", False, "a window is already running")
            return
        if health.state is not LinkState.CONNECTED:
            self._result("stability window", False, f"the link is {health.state.value}: connect first (the window measures a "
                                                    "CONNECTED link)")
            return
        self.meter.start(health, now)
        self._stability_confirmed = bool(operator_confirmed)
        self._stability_recorded = False
        self._stability_note = ""
        kind = "SIMULATED: it will be recorded as MOCK and counts for nothing" if self.simulated else "real link"
        self._result("stability window", True, f"started: {self.meter.window_s:.0f} s ({kind})")

    def _do_stability_cancel(self, now: float) -> None:
        self._cancel_meter("cancelled by the operator")

    def _cancel_meter(self, why: str) -> None:
        if self.meter.state == StabilityMeter.RUNNING:
            self.meter.reset()
            self._stability_note = f"cancelled: {why} (nothing recorded)"
            self._log(f"stability window cancelled: {why}")

    def _feed_meter(self, now: float) -> None:
        meter = self.meter
        if meter.state != StabilityMeter.RUNNING:
            return
        health = self.link.health(now)
        state = meter.feed(health, now)
        if state in (StabilityMeter.PASSED, StabilityMeter.FAILED) and not self._stability_recorded:
            self._stability_recorded = True
            outcome = self.acceptance.record_stability(meter, firmware_id=health.firmware_id, sketch_sha256=self._sketch_value(),
                                                       operator_confirmed=self._stability_confirmed, now_unix=self._unix(),
                                                       capabilities=self.capabilities)
            self._summary = None
            self._stability_note = f"{state}: {meter.reason}. Record: {outcome.status}: {outcome.note}"
            self._log(f"stability window {state}: {meter.reason}; recorded as {outcome.status} ({meter.evidence}): {outcome.note}")

    # ------------------------------------------------------------ events

    def _emit(self, signal: Any, value: Any) -> None:
        try:
            signal.emit(value)
        except RuntimeError:                                          # the QObject is gone (application closing)
            pass

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime(self._unix()))
        line = f"{stamp}  {text}"
        with self._lock:
            self._events.append(line)
        self._emit(self.event_added, line)

    @staticmethod
    def _new_items(items: list, seen: Any) -> list:
        if seen is None:
            return items
        for index in range(len(items) - 1, -1, -1):
            if items[index] == seen:
                return items[index + 1:]
        return items

    def _observe(self, now: float) -> None:
        link = self.link
        state = link.state
        if state is not self._last_state:
            self._dirty = True
            self._log(f"link {self._last_state.value} -> {state.value}")
            self._last_state = state
            if state is LinkState.CONNECTED and not self._connected_logged:
                self._connected_logged = True
                self._on_connected(now)
            if state in (LinkState.DISCONNECTED, LinkState.FAILED):
                self._connected_logged = False
        error = (link.error_kind, link.error)
        if error != self._last_error:
            self._last_error = error
            if link.error:
                self._log(f"link problem: {link.error}")
                if link.hint:
                    self._log(f"what to try: {link.hint}")
        latch = list(link.latch_log)
        for text in self._new_items(latch, self._latch_seen):
            self._log(f"link: {text}")
        self._latch_seen = latch[-1] if latch else self._latch_seen
        events = list(self.manager.events)
        for entry in self._new_items(events, self._manager_seen):
            text = entry[1]
            last_text, last_at = self._last_logged_text
            if text == last_text and now - last_at < 1.0:
                continue                                              # a repeated line (e.g. a persistent fault) is not a flood
            self._last_logged_text = (text, now)
            self._dirty = True
            self._log(f"mode: {text}")
        self._manager_seen = events[-1] if events else self._manager_seen

    def _on_connected(self, now: float) -> None:
        health = self.link.health(now)
        self._log(f"connected to {health.endpoint.label}: firmware {health.firmware_id or 'unknown'}; protocol "
                  f"{health.protocol_note}" + ("  [SIMULATED]" if self.simulated else ""))
        if self.simulated or self.store is None:
            return
        stamp = datetime.fromtimestamp(self._unix(), timezone.utc).isoformat()
        try:
            self.store.update_section(self.profile, "wifi", {"last_endpoint_ok_at": stamp})
        except Exception as error:
            self._log(f"could not remember when the endpoint last answered ({error})")

    # ------------------------------------------------------------ snapshots

    def _acceptance_summary(self, health: LinkHealth, now: float) -> AcceptanceSummary:
        key = (health.firmware_id, self._sketch_value())
        cached = self._summary
        if cached is not None and cached[0] == key and now - cached[1] < SUMMARY_PERIOD_S:
            return cached[2]
        summary = self.acceptance.summary(health.firmware_id, key[1], self._unix(), self.capabilities)
        self._summary = (key, now, summary)
        return summary

    def _stability_state(self, now: float) -> StabilityState:
        meter = self.meter
        return StabilityState(meter.state, meter.reason, meter.progress_s(now), meter.window_s, meter.evidence, meter.samples,
                              self._stability_note)

    def _build_snapshot(self, now: float) -> ServiceSnapshot:
        health = self.link.health(now)
        mode = self.manager.snapshot(now)
        plans = tuple(self.manager.plans[-20:])
        recent = [plan for plan in plans if plan.sent]
        last_age = None if not recent else max(0.0, now - recent[-1].t)
        transmitting = any(plan.sent and (plan.transmitted_throttle or plan.transmitted_steer) and now - plan.t <= TRANSMITTING_WINDOW_S
                           for plan in plans)
        with self._lock:
            events = tuple(self._events)[-SNAPSHOT_EVENTS:]
            request, discovery, discovering = self._last_request, self._discovery, self._discovering
        thread = self._thread
        return ServiceSnapshot(
            t=now, unix=self._unix(), running=thread is not None and thread.isRunning(), simulated=self.simulated,
            mock_session=self.mock_session, health=health, mode=mode, gate=mode.gate, authority=mode.authority,
            controller=self.link.controller(now), acceptance=self._acceptance_summary(health, now),
            stability=self._stability_state(now), plans=plans, events=events, ever_sent=self.link.drive_command_ever_sent,
            transmitting=transmitting, last_command_age_s=last_age, last_request=request, discovery=discovery,
            discovering=discovering, endpoint=self.link.endpoint, preferred_ssid=self._prefs.preferred_ssid,
            store_note=self._prefs.problem, worker_error=self._worker_error)

    def _publish(self, now: float) -> None:
        self._dirty = False
        if self._worker_error and time.monotonic() - self._worker_error_at > ERROR_CLEAR_S:
            self._worker_error = None
        snapshot = self._build_snapshot(now)
        self._latest = snapshot
        self._last_snapshot_at = now
        self._emit(self.snapshot_changed, snapshot)
