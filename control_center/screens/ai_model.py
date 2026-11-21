"""AI / MODEL: which policy file this is, whether it is loaded, what it is fed and whether that is real, what it decides and what a
decision would become on the wire. Display and software checks only: nothing on this screen can send anything to the robot.

* The model is loaded ONCE (``policy_engine`` caches it per process) on a worker thread; a second load while one is running is
  refused. Nothing loads at start-up unless a model path is remembered for the profile, and even then it only LOADS: no
  inference loop starts by itself.
* The 19 observation values are shown with the live status of each group (IMU, previous action, own pose, opponent): VALID /
  MISSING / STALE, age, source LIVE / SIMULATED / REPLAY / CONSTANT, and the reason. The baseline firmware sends no IMU, so a
  complete LIVE observation cannot exist and ``readiness()`` cannot be READY. Nothing is invented to fill a gap.
* The inference LOOP is owned here but started only by a deliberate Start Round on the MATCH tab (``start_loop`` registers with the
  engine's single-loop registry; ``shutdown`` releases it). ``Diagnostic inference`` runs the model once on a SIMULATED
  observation and is never valid for control.
* The mapping self-check is a SOFTWARE check; it feeds ``MatchInputs.mapping_validated`` and is never hardware evidence.

Public API used by the MATCH tab: ``model_loaded()``, ``mapping_validated()``, ``last_decision()``, ``observation_status()``,
``start_loop()`` / ``stop_loop()`` / ``infer_once(now)``, ``last_inference_error()``, ``readiness()``.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from control_center.backend import match_support as ms
from control_center.backend.command_mapping import FIRMWARE_PC_LIMIT, LIMITS, describe_mapping
from control_center.backend.observation_builder import (SOURCE_LIVE, SOURCE_SIMULATED, ImuInput, ObservationInputs, OpponentInput,
                                                        OwnPoseInput, build_observation, default_contract)
from control_center.backend.policy_engine import DuplicateLoopError, engine_for
from control_center.backend.readiness import CHIP_COLOURS, NOT_CONFIGURED, Readiness
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT, load_contract
from control_center.widgets.panel_kit import (STYLE, KeyValueGrid, fill_table, label, make_table, panel, set_text, table_texts)

log = logging.getLogger(__name__)
TITLE = "AI / MODEL"
REFRESH_MS = 500
ROOT = Path(__file__).resolve().parents[2]
WORD_LEVEL = {"VALID": "ok", "MISSING": "bad", "STALE": "warn"}
DIAGNOSTIC_OWN = (0.575, 0.575, 0.0)          # arena centre, nose along +x (a simulated pose for the diagnostic inference)
DIAGNOSTIC_OPPONENT = (0.9, 0.3)
DIAGNOSTIC_NOTE = ("DIAGNOSTIC inference: the model is run once on a SIMULATED observation (own robot at the arena centre facing +x, "
                   "opponent at (0.90, 0.30) m, an IMU at rest: gyro 0, accelerometer (0, 0, +9.81)). It checks that the model "
                   "loads and answers. It is never valid for control, is never sent anywhere, and says nothing about the real robot.")


class _Job(QObject):
    """One function run on a daemon thread; ``done((job, (result, error_text)))`` arrives on the GUI thread. Not a QThread: a load cannot be
    interrupted and a destroyed running QThread aborts the process, a daemon thread just ends with it."""

    done = Signal(object)

    def __init__(self, function: Callable[[], Any], kind: str, engine: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._function = function
        self.kind, self.engine = kind, engine
        self._abandoned = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="ai-model-job", daemon=True)
        self.thread.start()

    def abandon(self) -> None:
        self._abandoned = True

    def _run(self) -> None:
        try:
            outcome: tuple[Any, str] = (self._function(), "")
        except Exception as error:
            log.exception("model job failed")
            outcome = (None, f"{type(error).__name__}: {error}")
        if not self._abandoned:
            try:
                self.done.emit((self, outcome))
            except RuntimeError:                  # the screen was deleted meanwhile
                pass


class AiModelScreen(QWidget):
    def __init__(self, hub: Any, parent: QWidget | None = None, *, tracking: Any = None,
                 engine_factory: Callable[[str], Any] | None = None, clock: Callable[[], float] = time.monotonic,
                 autoload: bool = True, start_timer: bool = True) -> None:
        super().__init__(parent)
        self.hub = hub
        self.tracking = tracking
        self._engine_factory = engine_factory or (lambda path: engine_for(path))
        self._clock = clock
        self.setWindowTitle(TITLE)
        self.setStyleSheet(STYLE)
        self._closed = False
        self._state = "not_loaded"                   # not_loaded | loading | loaded | failed
        self._info: Any = None
        self._load_error = ""
        self._load_started = 0.0
        self._load_job: _Job | None = None
        self._info_job: _Job | None = None
        self._load_requests = 0
        self._message = ""
        self._token: Any = None
        self._contracts: dict[str, Any] = {}
        self._decision: Any = None
        self._decision_at: float | None = None
        self._decision_source = ""
        self._inference_error = ""
        self._view: ms.ObservationView | None = None
        self._view_live: ms.LiveInputs | None = None
        self._diagnostic: Any = None
        self._diagnostic_text = ""
        self._mapping: ms.MappingCheck | None = None
        self._path = str(ms.DEFAULT_MODEL_PATH)
        self._engine: Any = None

        remembered = self._remembered_path()
        if remembered:
            self._path = remembered
        self._build()
        self._engine = self._engine_factory(self.resolved_path())
        if getattr(self._engine, "loaded", False):
            self._state = "loaded"                   # already in this process's cache (a restarted screen): reuse, do not reload
        self._start_info_job()
        if hasattr(hub, "profile_changed"):
            try:
                hub.profile_changed.connect(self._on_profile_changed)
            except Exception:
                pass
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        if start_timer:
            self._timer.start()
        self.paint_all()
        if remembered and autoload and Path(self.resolved_path()).is_file() and self._state != "loaded":
            self.load_model()                        # ONLY a load. No inference loop is ever started from here.

    # ================================================================ construction

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        header = QHBoxLayout()
        title = label(TITLE, "screenTitle", selectable=False)
        title.setWordWrap(False)
        header.addWidget(title)
        header.addStretch(1)
        self.profile_label = label("", "hint")
        header.addWidget(self.profile_label)
        self.readiness_chip = QLabel(NOT_CONFIGURED)
        self.readiness_chip.setObjectName("chip")
        header.addWidget(self.readiness_chip)
        outer.addLayout(header)
        self.evidence_label = label("Evidence on this tab is SOFTWARE or SIMULATED. Nothing here is a hardware result, and nothing here "
                                    "sends anything to the robot.", "hint")
        outer.addWidget(self.evidence_label)

        body = QWidget()
        body.setObjectName("scrollBody")
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 6, 0)
        column.setSpacing(8)

        self.origin_label = label("", "banner")
        self.origin_label.setStyleSheet("background: #4d3a12; color: #ffeab5;")
        column.addWidget(self.origin_label)

        frame, box = panel("MODEL")
        row = QHBoxLayout()
        row.addWidget(QLabel("Checkpoint (.zip):"))
        self.path_edit = QLineEdit(self._path)
        self.path_edit.setToolTip("Only *.zip checkpoint files. Press Enter or Use this path to select it.")
        self.path_edit.returnPressed.connect(lambda: self.set_model_path(self.path_edit.text()))
        row.addWidget(self.path_edit, 1)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.browse_button)
        self.use_path_button = QPushButton("Use this path")
        self.use_path_button.clicked.connect(lambda: self.set_model_path(self.path_edit.text()))
        row.addWidget(self.use_path_button)
        self.default_button = QPushButton("Use default")
        self.default_button.clicked.connect(lambda: self.set_model_path(ms.DEFAULT_MODEL_PATH))
        row.addWidget(self.default_button)
        box.addLayout(row)
        self.card = KeyValueGrid()
        box.addWidget(self.card)
        row = QHBoxLayout()
        self.load_button = QPushButton("Load model")
        self.load_button.setToolTip("Loads the checkpoint on a background thread (once per process; a loaded model is reused).")
        self.load_button.clicked.connect(lambda: self.load_model())
        row.addWidget(self.load_button)
        self.load_status = label("", "headline")
        row.addWidget(self.load_status, 1)
        box.addLayout(row)
        self.message_label = label("", "hint")
        box.addWidget(self.message_label)
        column.addWidget(frame)

        frame, box = panel("OBSERVATION: the 19 values the model reads, and whether each group is real")
        box.addWidget(label(ms.IMU_BLOCK_TEXT, "hint"))
        box.addWidget(label(ms.CONSTANT_TEXT, "hint"))
        self.observation_headline = label("", "headline")
        box.addWidget(self.observation_headline)
        self.group_table = make_table(["Group", "Channels", "Status", "Age", "Source", "Reason"])
        self.group_table.setMinimumHeight(150)
        box.addWidget(self.group_table)
        self.channel_table = make_table(["#", "Channel", "Group", "Value", "Status", "Age", "Source"])
        self.channel_table.setMinimumHeight(510)
        box.addWidget(self.channel_table)
        self.observation_notes = label("", "hint")
        box.addWidget(self.observation_notes)
        column.addWidget(frame)

        frame, box = panel("CURRENT DECISION")
        self.decision_headline = label("No decision yet", "headline")
        box.addWidget(self.decision_headline)
        self.decision_grid = KeyValueGrid()
        box.addWidget(self.decision_grid)
        box.addWidget(label("A decision is data. It becomes a command only through command mapping and the mode manager's gated "
                            "sender, which needs hardware acceptance that does not exist yet.", "hint"))
        column.addWidget(frame)

        frame, box = panel("COMMAND MAPPING (tactic to throttle / steer)")
        self.limits_grid = KeyValueGrid()
        box.addWidget(self.limits_grid)
        self.mapping_table = make_table(["Tactic", "Throttle", "Steer", "Left drive", "Right drive", "Meaning"])
        self.mapping_table.setMinimumHeight(190)
        box.addWidget(self.mapping_table)
        box.addWidget(label(ms.MOTION_CHECK_NOTE, "hint"))
        box.addWidget(label(ms.DIRECTION_NOTE, "hint"))
        row = QHBoxLayout()
        self.check_button = QPushButton("Check mapping")
        self.check_button.setToolTip("Software self-check of every action, the caps and invalid input. No robot involved.")
        self.check_button.clicked.connect(lambda: self.check_mapping())
        row.addWidget(self.check_button)
        self.check_result = label("Not run. SOFTWARE self-check, never a hardware result.", "headline")
        row.addWidget(self.check_result, 1)
        box.addLayout(row)
        self.check_details = label("", "hint")
        box.addWidget(self.check_details)
        column.addWidget(frame)

        frame, box = panel("DIAGNOSTIC INFERENCE (SIMULATED input)")
        box.addWidget(label(DIAGNOSTIC_NOTE, "hint"))
        row = QHBoxLayout()
        self.diagnostic_button = QPushButton("Diagnostic inference")
        self.diagnostic_button.clicked.connect(lambda: self.diagnostic_inference())
        row.addWidget(self.diagnostic_button)
        row.addStretch(1)
        box.addLayout(row)
        self.diagnostic_label = label("", "headline")
        box.addWidget(self.diagnostic_label)
        column.addWidget(frame)
        column.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

    # ================================================================ paths and the remembered preference

    def profile_name(self) -> str:
        return str(getattr(self.hub, "profile_name", "live"))

    def resolved_path(self) -> str:
        path = Path(self._path)
        return str(path if path.is_absolute() else ROOT / path)

    def model_path(self) -> str:
        return self._path

    def _remembered_path(self) -> str | None:
        store = getattr(self.hub, "settings_store", None)
        if store is None:
            return None
        try:
            value = store.get_section(self.profile_name(), "preferences").get("model_path")
        except Exception:
            return None
        return value if isinstance(value, str) and value.strip() else None

    def _remember(self, path: str) -> str:
        store = getattr(self.hub, "settings_store", None)
        if store is None:
            return "not remembered: this session keeps no settings"
        try:
            result = store.update_section(self.profile_name(), "preferences", {"model_path": path})
        except Exception as error:
            return f"not remembered: {error}"
        return f"remembered for profile '{self.profile_name()}'" if result.ok else f"not remembered: {result.error}"

    def _on_profile_changed(self, _name: str = "") -> None:
        remembered = self._remembered_path()
        if remembered and remembered != self._path and self._state != "loading" and self._token is None:
            self.set_model_path(remembered, remember=False)

    def _browse(self) -> None:
        start = str(Path(self.resolved_path()).parent)
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose a checkpoint", start, "Checkpoint (*.zip)")
        if chosen:
            self.set_model_path(chosen)

    def set_model_path(self, text: str, *, remember: bool = True) -> tuple[bool, str]:
        """Select another checkpoint file (*.zip only). Refused while loading or while an inference loop runs."""
        text = str(text).strip().strip('"')
        if self._state == "loading":
            return self._say(False, "a model is loading: wait for it to finish before choosing another file")
        if self._token is not None:
            return self._say(False, "a round is using the model: stop the round before choosing another file")
        if not text.lower().endswith(".zip"):
            return self._say(False, "only *.zip checkpoint files can be chosen")
        self._path = text
        self.path_edit.setText(text)
        self._engine = self._engine_factory(self.resolved_path())
        self._state = "loaded" if getattr(self._engine, "loaded", False) else "not_loaded"
        self._info, self._load_error = None, ""
        self._decision, self._decision_at, self._inference_error = None, None, ""
        self._contracts.pop(self.resolved_path(), None)
        self._start_info_job()
        note = self._remember(text) if remember else ""
        self._say(True, f"checkpoint set to {text}" + (f" ({note})" if note else ""))
        self.paint_all()
        return True, self._message

    def _say(self, ok: bool, text: str) -> tuple[bool, str]:
        self._message = text
        if hasattr(self, "message_label"):
            set_text(self.message_label, text, "ok" if ok else "bad")
        return ok, text

    # ================================================================ identity and loading (worker thread)

    def engine(self) -> Any:
        return self._engine

    def _start_info_job(self) -> None:
        engine = self._engine
        if engine is None:
            return
        if self._info_job is not None:
            self._info_job.abandon()
        job = _Job(engine.info, "info", engine, self)
        job.done.connect(self._on_job_done)
        self._info_job = job
        job.start()

    def _on_job_done(self, payload: tuple[_Job, tuple[Any, str]]) -> None:
        job, outcome = payload
        if job.kind == "info":
            self._on_info(job, outcome)
        else:
            self._on_loaded(job, job.engine, outcome)

    def _on_info(self, job: _Job, outcome: tuple[Any, str]) -> None:
        if job is not self._info_job or self._closed:
            return
        info, error = outcome
        if info is not None and self._info is None:
            self._info = info
        elif error:
            self._load_error = error
        self.paint_all()

    def load_model(self) -> bool:
        """Load the checkpoint on a worker thread. A second request while one runs is refused; a loaded model is reused."""
        if self._state == "loading":
            self._say(False, "a load is already running: it is refused to start a second one")
            return False
        if self._state == "loaded" and getattr(self._engine, "loaded", False):
            self._say(True, "the model is already loaded: it is reused, not loaded again")
            return True
        engine = self._engine
        self._state, self._load_error, self._load_started = "loading", "", self._clock()
        self._load_requests += 1
        job = _Job(engine.load, "load", engine, self)
        job.done.connect(self._on_job_done)
        self._load_job = job
        self._say(True, "loading on a background thread (usually 30 to 60 s); the rest of the screen keeps working")
        job.start()
        self.paint_all()
        return True

    def _on_loaded(self, job: _Job, engine: Any, outcome: tuple[Any, str]) -> None:
        if job is not self._load_job or self._closed or engine is not self._engine:
            return
        info, error = outcome
        if error:
            self._state, self._load_error = "failed", error
        elif not getattr(engine, "loaded", False):
            problems = "; ".join(getattr(info, "problems", ()) or ()) or str(getattr(engine, "load_error", "") or "the model did not load")
            self._state, self._load_error, self._info = "failed", problems, info or self._info
        else:
            self._state, self._load_error, self._info = "loaded", "", info
        self._load_job = None
        self._say(self._state == "loaded", "model loaded" if self._state == "loaded" else f"the model did not load: {self._load_error}")
        self.paint_all()

    def model_loaded(self) -> bool:
        return self._state == "loaded" and bool(getattr(self._engine, "loaded", False))

    def model_state(self) -> str:
        return self._state

    def model_info(self) -> Any:
        return self._info

    def model_detail(self) -> str:
        if self.model_loaded():
            return f"policy loaded ({Path(self._path).name})"
        return {"loading": "policy is loading", "failed": f"policy failed to load: {self._load_error}"}.get(self._state, "policy not loaded")

    # ================================================================ the inference loop (owned here, driven by the MATCH tab)

    @property
    def loop_active(self) -> bool:
        return self._token is not None

    def start_loop(self, owner: str = "AI / MODEL screen (round)") -> tuple[bool, str]:
        """Register THE inference loop with the engine. A second one anywhere in the process is refused."""
        if self._token is not None:
            return True, "the loop is already running"
        if not self.model_loaded():
            return False, "the model is not loaded"
        try:
            self._token = self._engine.start(owner)
        except DuplicateLoopError as error:
            return False, str(error)
        self._decision, self._decision_at, self._inference_error = None, None, ""
        return True, "inference loop registered"

    def stop_loop(self) -> None:
        token, self._token = self._token, None
        if token is not None:
            try:
                self._engine.stop(token)
            except Exception:
                log.exception("releasing the inference loop failed")

    def contract(self) -> Any:
        path = self.resolved_path()
        if Path(path) == RECOMMENDED_CHECKPOINT:
            return default_contract()
        if path not in self._contracts:
            self._contracts[path] = load_contract(path)
        return self._contracts[path]

    def observation_status(self, now: float | None = None) -> ms.ObservationView:
        """The 19-value observation as it can be built right now from the latest tracking result and telemetry. Never raises."""
        now = self._clock() if now is None else now
        try:
            contract = self.contract()
        except Exception as error:
            contract = None
            view, live = ms.describe_observation(None, None, error=f"the checkpoint's observation contract could not be read: {error}"), None
            self._view, self._view_live = view, live
            return view
        view, live = ms.observation_view(self.hub, self.tracking, now, contract)
        self._view, self._view_live = view, live
        return view

    def live_inputs(self) -> ms.LiveInputs | None:
        return self._view_live

    def infer_once(self, now: float | None = None) -> Any:
        """One inference on the current observation. Only while the loop is registered and the model loaded; never blocks on a load.
        Returns the PolicyDecision (blocked decisions included) or None when nothing could be asked."""
        now = self._clock() if now is None else now
        if self._token is None:
            return None
        if not self.model_loaded():
            self._inference_error = "the model is not loaded"
            return None
        view = self.observation_status(now)
        if view.result is None:
            self._inference_error = view.error or "no observation could be built"
            return None
        try:
            decision = self._engine.decide(view.result, now)
        except Exception as error:                          # the engine does not raise; a fake or a bug must not kill the tick
            log.exception("decide failed")
            self._inference_error = f"{type(error).__name__}: {error}"
            return None
        self._decision, self._decision_at, self._decision_source = decision, now, "inference loop"
        self._inference_error = getattr(decision, "error", "") or ""
        return decision

    def last_decision(self) -> Any:
        return self._decision

    def last_decision_at(self) -> float | None:
        return self._decision_at

    def last_inference_error(self) -> str:
        return self._inference_error

    def last_diagnostic(self) -> Any:
        return self._diagnostic

    # ================================================================ diagnostic inference (SIMULATED input)

    def diagnostic_inference(self, now: float | None = None) -> Any:
        """Run the model once on a SIMULATED observation. DIAGNOSTIC: ``valid_for_control`` is always False. Needs a loaded model."""
        now = self._clock() if now is None else now
        if not self.model_loaded():
            self._diagnostic, self._diagnostic_text = None, "the model is not loaded: press Load model first"
            self.paint_all()
            return None
        x, y, heading = DIAGNOSTIC_OWN
        inputs = ObservationInputs(SOURCE_SIMULATED, OwnPoseInput(x, y, heading, now, "measured"),
                                   OpponentInput(DIAGNOSTIC_OPPONENT[0], DIAGNOSTIC_OPPONENT[1], now, "tracked"),
                                   ImuInput((0.0, 0.0, 0.0), (0.0, 0.0, 9.81), now, SOURCE_SIMULATED, True))
        try:
            result = build_observation(inputs, now, self.contract())
            decision = self._engine.decide(result, now)
        except Exception as error:
            self._diagnostic, self._diagnostic_text = None, f"diagnostic inference failed: {type(error).__name__}: {error}"
            self.paint_all()
            return None
        self._diagnostic = decision
        if decision.tactic is None:
            self._diagnostic_text = "DIAGNOSTIC: no tactic. " + "; ".join(decision.blocked_by or (decision.error,))
        else:
            self._diagnostic_text = (f"DIAGNOSTIC (SIMULATED input, never valid for control): tactic {decision.tactic.upper()}"
                                     + ("" if decision.probability is None else f", probability {decision.probability:.2f}")
                                     + f", source {decision.source}, valid_for_control {decision.valid_for_control}")
        self.paint_all()
        return decision

    # ================================================================ the mapping self-check (SOFTWARE)

    def check_mapping(self) -> ms.MappingCheck:
        self._mapping = ms.mapping_self_check(LIMITS)
        self.paint_all()
        return self._mapping

    def mapping_validated(self) -> bool:
        return bool(self._mapping is not None and self._mapping.passed and self._mapping.limits == LIMITS)

    def mapping_check(self) -> ms.MappingCheck | None:
        return self._mapping

    def mapping_rows(self) -> list[dict[str, Any]]:
        return describe_mapping(LIMITS)

    # ================================================================ readiness

    def readiness(self) -> Readiness:
        return ms.ai_readiness(state=self._state, info=self._info, error=self._load_error if self._state == "failed" else "",
                               view=self.observation_status())

    # ================================================================ painting

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self.paint_all()

    def refresh(self) -> None:
        if self._closed:
            return
        visible = self.isVisible()
        if visible:
            ms.ensure_tracking_active(self.tracking)
        view = self.observation_status()
        readiness = ms.ai_readiness(state=self._state, info=self._info, error=self._load_error if self._state == "failed" else "", view=view)
        text = f"{readiness.level.upper()}: {readiness.headline}"
        if self.readiness_chip.text() != text:
            self.readiness_chip.setText(text)
            self.readiness_chip.setStyleSheet(f"background: {CHIP_COLOURS[readiness.level]};")
            self.readiness_chip.setToolTip(readiness.detail + ("\n" + readiness.fix if readiness.fix else ""))
        set_text(self.profile_label, f"profile: {self.profile_name()}")
        self._paint_card()
        if not visible and not getattr(self, "_force_paint", False):
            return
        self._paint_observation(view)
        self._paint_decision()
        self._paint_mapping()
        self.diagnostic_label.setText(self._diagnostic_text)
        self.load_button.setEnabled(self._state != "loading")
        self.check_button.setEnabled(True)
        self.diagnostic_button.setEnabled(self.model_loaded())

    def paint_all(self) -> None:
        """Paint every panel even while the tab is hidden (tests and screenshots)."""
        self._force_paint = True
        try:
            self.refresh()
        finally:
            self._force_paint = False

    def _paint_card(self) -> None:
        info = self._info
        audited = Path(self.resolved_path()) == RECOMMENDED_CHECKPOINT
        set_text(self.origin_label, ms.MODEL_ORIGIN_TEXT if audited else ms.MODEL_UNAUDITED_TEXT)
        rows: list[tuple[str, str, str]] = [("Path", self.resolved_path(), "plain")]
        if info is None:
            rows.append(("Identity", "reading the checkpoint record...", "off"))
        else:
            rows += [("SHA-256", info.sha256 or "(unreadable)", "plain"),
                     ("Size", f"{info.size_bytes:,} bytes", "plain"),
                     ("Trained with", f"stable-baselines3 {info.sb3_version_trained or 'unknown'}, torch {info.torch_version_trained or 'unknown'}", "plain"),
                     ("Installed here", f"stable-baselines3 {info.sb3_version_installed or 'NOT INSTALLED'}, torch {info.torch_version_installed or 'NOT INSTALLED'}",
                      "plain" if info.sb3_version_installed and info.torch_version_installed else "bad"),
                     ("Observation space", f"Box(-1, 1, ({info.obs_dim},), float32)  profile {info.observation_profile or 'unknown'}, "
                                           f"previous action {info.previous_action_encoding or 'unknown'}", "plain"),
                     ("Action space", f"{info.action_space}: " + (", ".join(f"{i} {t}" for i, t in enumerate(info.tactics)) or "tactics unknown")
                      + f"  (profile {info.action_profile or 'unknown'}; deterministic prediction {info.deterministic})", "plain")]
            verdict = "COMPATIBLE" if info.compatible else "INCOMPATIBLE"
            rows.append(("Compatibility", verdict + (": " + "; ".join(info.problems) if info.problems else ""), "ok" if info.compatible else "bad"))
            if info.warnings:
                rows.append(("Warnings", "; ".join(info.warnings), "warn"))
        loaded = self.model_loaded()
        seconds = getattr(self._info, "load_seconds", None) if self._info is not None else None
        if loaded:
            word = "LOADED"
            state = "LOADED" + ("" if seconds is None else f" in {seconds:.1f} s") + "; loaded once and reused for every round"
        elif self._state == "loading":
            word = state = f"LOADING: {max(0.0, self._clock() - self._load_started):.0f} s elapsed (usually 30 to 60 s)"
        elif self._state == "failed":
            word, state = "LOAD FAILED", f"LOAD FAILED: {self._load_error}"
        else:
            word = state = "NOT LOADED"
        level = "ok" if loaded else ("bad" if self._state == "failed" else "warn")
        rows.append(("Model state", state, level))
        self.card.set_rows(rows)
        set_text(self.load_status, word, level)

    def _paint_observation(self, view: ms.ObservationView) -> None:
        set_text(self.observation_headline, f"{view.label}: {view.statement}", "ok" if view.valid_for_control else ("warn" if view.complete else "bad"))
        fill_table(self.group_table, [([r.label, r.indices, r.word, "-" if r.age_ms is None else f"{r.age_ms:.0f} ms", r.source, r.reason],
                                       WORD_LEVEL.get(r.word, "plain") if r.source != "CONSTANT" else "off") for r in view.groups])
        fill_table(self.channel_table, [([str(c.index), c.name, c.group, "-" if c.value is None else f"{c.value:+.4f}" + (" (constant)" if c.constant else ""),
                                          c.word, "-" if c.age_ms is None else f"{c.age_ms:.0f} ms", c.source],
                                         WORD_LEVEL.get(c.word, "plain") if not c.constant else "off") for c in view.channels])
        live = self._view_live
        notes = list(view.notes) + (list(live.notes) if live is not None else [])
        blocked = [b for b in view.blocked_by if not b.startswith("[")]
        set_text(self.observation_notes, ("Blocked for control: " + "; ".join(blocked) + "   " if blocked else "") + "   ".join(dict.fromkeys(notes)))

    def _paint_decision(self) -> None:
        decision, at = self._decision, self._decision_at
        if decision is None:
            decision, origin = self._diagnostic, "diagnostic inference button"
            if decision is None:
                set_text(self.decision_headline, "No decision yet", "off")
                self.decision_grid.set_rows([("Why", "no inference has run: the loop runs only during a round (MATCH tab), or press Diagnostic inference",
                                              "off")])
                return
        else:
            origin = self._decision_source or "inference loop"
        live = decision.valid_for_control
        age = "" if at is None or decision is self._diagnostic else f", decided {max(0.0, self._clock() - at):.1f} s ago" + (
            "" if self._token is not None else " (no loop running: this is NOT current)")
        set_text(self.decision_headline, (decision.tactic.upper() if decision.tactic else "NO TACTIC") + "  " +
                 (ms.LIVE_LABEL if live else ms.DIAGNOSTIC_LABEL), "ok" if live else "warn")
        self.decision_grid.set_rows([
            ("Tactic", decision.tactic or "none (blocked)", "plain"),
            ("Probability", "-" if decision.probability is None else f"{decision.probability:.2f}", "plain"),
            ("Source", f"{decision.source}  (from the {origin}{age})", "ok" if decision.source == SOURCE_LIVE else "warn"),
            ("valid_for_control", str(decision.valid_for_control), "ok" if live else "bad"),
            ("blocked_by", "; ".join(decision.blocked_by) or "nothing", "plain" if not decision.blocked_by else "bad"),
            ("Label", ms.LIVE_LABEL if live else ms.DIAGNOSTIC_LABEL, "ok" if live else "warn")])

    def _paint_mapping(self) -> None:
        rows = describe_mapping(LIMITS)
        fill_table(self.mapping_table, [([r["tactic"], str(r["throttle"]), str(r["steer"]), str(r["left"]), str(r["right"]), r["meaning"]],
                                         "plain") for r in rows])
        self.limits_grid.set_rows([
            ("max_command", f"{LIMITS.max_command} of 255 (about {100 * LIMITS.max_command / 255:.0f} %); the firmware's own limit is {FIRMWARE_PC_LIMIT}", "plain"),
            ("operating scale", f"{LIMITS.operating_scale:.3f} of the simulator's full-scale tactic command", "plain"),
            ("max decision age", f"{1000 * LIMITS.max_age_s:.0f} ms: an older decision becomes 0, 0", "plain"),
            ("slew limit", f"{LIMITS.slew_full_scale_s:.2f} s from 0 to max_command (decreases and stops are immediate)", "plain"),
            ("neutral-first", f"0, 0 for {1000 * LIMITS.reversal_neutral_s:.0f} ms when throttle or steer flips sign", "plain")])
        check = self._mapping
        if check is None:
            set_text(self.check_result, "Not run. SOFTWARE self-check, never a hardware result.", "off")
            self.check_details.setText("")
        else:
            set_text(self.check_result, check.summary(), "ok" if check.passed else "bad")
            self.check_details.setText("" if check.passed else "Failed: " + "; ".join(f"{l.name} ({l.detail})" for l in check.failures))

    # ================================================================ test helpers

    def group_texts(self) -> list[list[str]]:
        return table_texts(self.group_table)

    def channel_texts(self) -> list[list[str]]:
        return table_texts(self.channel_table)

    def mapping_texts(self) -> list[list[str]]:
        return table_texts(self.mapping_table)

    # ================================================================ shutdown

    def shutdown(self) -> None:
        """Stop the timer, release the inference loop and drop any job result still on its way. Closes nothing it does not own."""
        if self._closed:
            return
        self.stop_loop()
        self._closed = True
        self._timer.stop()
        for job in (self._load_job, self._info_job):
            if job is not None:
                job.abandon()
        if hasattr(self.hub, "profile_changed"):
            try:
                self.hub.profile_changed.disconnect(self._on_profile_changed)
            except (TypeError, RuntimeError):
                pass

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.shutdown()
        super().closeEvent(event)
