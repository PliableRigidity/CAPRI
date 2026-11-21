"""The one loaded copy of the trained checkpoint, and decisions made with it. No camera, port, socket or Qt.

* ``inspect_model(path)``   identity and compatibility from the zip only (sha256, size, library versions stored in it,
                            observation/action space) - cheap, never loads torch weights. Returns ``ModelInfo``.
* ``PolicyEngine``          loads the model ONCE per process (module-level cache keyed by path + mtime + sha256, with a
                            lock so two threads asking at once still load once), predicts DETERMINISTICALLY, serialises
                            inference with a lock, and refuses a second inference loop (``start`` / ``stop``).
* ``decide(result)``        ObservationResult -> ``PolicyDecision`` carrying ``source`` (LIVE / SIMULATED / REPLAY),
                            ``valid_for_control``, ``blocked_by`` and per-group ages. An inference error, an incompatible
                            model or an incomplete observation yields a BLOCKED decision (tactic None): never an exception
                            into the UI and never a previous action re-used.

Why deterministic: every evaluation and replay of this checkpoint in the repository calls ``predict(..., deterministic=True)``
(autobot/sim/cli.py:313, :557; scripts/reactive_recovery.py:376, :499), and the training record says the weights were fitted
by supervised imitation, not sampled PPO rollouts (runs/recovery-challenger-v2-seed1/effective_config.json:3).

Nothing here sends anything: a decision is data. ``valid_for_control`` is the ONLY flag a downstream command path may honour,
and it is False for any input that is not a real, fresh, LIVE measurement.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from control_center.backend.observation_builder import ObservationResult
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT, ContractError, load_contract

log = logging.getLogger(__name__)

DETERMINISTIC = True
OBSERVATION_DIM = 19                      # runs/recovery-challenger-v2-seed1/final_model.zip, observation_space._shape
SB3_VERSION_RECORD = "_stable_baselines3_version"
SYSTEM_INFO_RECORD = "system_info.txt"


class DuplicateLoopError(RuntimeError):
    """A second inference loop was requested while one is running."""


# ----------------------------------------------------------------------------------------------
# Identity and compatibility (zip only)


@dataclass(frozen=True)
class ModelInfo:
    path: str
    sha256: str
    size_bytes: int
    sb3_version_trained: str | None
    sb3_version_installed: str | None
    torch_version_trained: str | None
    torch_version_installed: str | None
    obs_dim: int | None
    action_space: str                       # e.g. "Discrete(5)"
    tactics: tuple[str, ...]
    observation_profile: str | None
    action_profile: str | None
    previous_action_encoding: str | None
    deterministic: bool
    loaded: bool
    loaded_at: float | None                 # time.time() of the load, None until loaded
    load_seconds: float | None
    compatible: bool
    problems: tuple[str, ...]               # anything here blocks inference
    warnings: tuple[str, ...] = ()          # worth knowing, does not block

    def to_report(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes,
                "sb3_version_trained": self.sb3_version_trained, "sb3_version_installed": self.sb3_version_installed,
                "torch_version_trained": self.torch_version_trained, "torch_version_installed": self.torch_version_installed,
                "obs_dim": self.obs_dim, "action_space": self.action_space, "tactics": list(self.tactics),
                "observation_profile": self.observation_profile, "action_profile": self.action_profile,
                "previous_action_encoding": self.previous_action_encoding, "deterministic": self.deterministic,
                "loaded": self.loaded, "loaded_at": self.loaded_at, "load_seconds": self.load_seconds,
                "compatible": self.compatible, "problems": list(self.problems), "warnings": list(self.warnings)}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_versions() -> tuple[str | None, str | None]:
    try:
        import stable_baselines3
        sb3 = str(stable_baselines3.__version__)
    except Exception:
        sb3 = None
    try:
        import torch
        torch_version = str(torch.__version__)
    except Exception:
        torch_version = None
    return sb3, torch_version


def _system_info(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.lstrip("- ").partition(":")
        if value:
            out[key.strip()] = value.strip()
    return out


_INFO_CACHE: dict[tuple[str, int, int], tuple[str, ModelInfo]] = {}
_INFO_LOCK = threading.Lock()


def inspect_model(path: Path | str = RECOMMENDED_CHECKPOINT) -> ModelInfo:
    """Identity and compatibility of a checkpoint from the zip and the installed libraries. Never loads the model."""
    file = Path(path)
    if file.suffix != ".zip" and not file.is_file():
        file = file.with_name(file.name + ".zip")
    file = file.resolve()
    sb3, torch_version = _installed_versions()
    problems: list[str] = []
    warnings: list[str] = []
    try:
        stat = file.stat()
    except OSError as error:
        return ModelInfo(str(file), "", 0, None, sb3, None, torch_version, None, "unknown", (), None, None, None, DETERMINISTIC,
                         False, None, None, False, (f"checkpoint file unreadable: {error}",))
    key = (str(file), stat.st_mtime_ns, stat.st_size)
    with _INFO_LOCK:
        cached = _INFO_CACHE.get(key)
    if cached:
        return cached[1]
    sha = sha256_of(file)

    trained_sb3 = trained_torch = None
    obs_dim = None
    action_space = "unknown"
    tactics: tuple[str, ...] = ()
    profile = action_profile = encoding = None
    try:
        with zipfile.ZipFile(file) as archive:
            trained_sb3 = archive.read(SB3_VERSION_RECORD).decode().strip() if SB3_VERSION_RECORD in archive.namelist() else None
            system = _system_info(archive.read(SYSTEM_INFO_RECORD).decode()) if SYSTEM_INFO_RECORD in archive.namelist() else {}
            data = json.loads(archive.read("data"))
        trained_torch = system.get("PyTorch")
        shape = (data.get("observation_space") or {}).get("_shape") or []
        obs_dim = int(shape[0]) if len(shape) == 1 else None
        n = (data.get("action_space") or {}).get("n")
        action_space = f"Discrete({int(n)})" if n is not None else "unknown"
        embedded = data.get("autobot_observation") if isinstance(data.get("autobot_observation"), dict) else {}
        profile, action_profile, encoding = (embedded.get(k) for k in ("observation_profile", "action_profile", "previous_action_encoding"))
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as error:
        problems.append(f"not a readable Stable-Baselines3 checkpoint zip: {type(error).__name__}: {error}")

    if sb3 is None:
        problems.append("stable-baselines3 is not installed in this environment: the model cannot be loaded")
    elif trained_sb3 is None:
        warnings.append("the zip does not record the stable-baselines3 version that saved it")
    elif trained_sb3 != sb3:
        (problems if trained_sb3.split(".")[0] != sb3.split(".")[0] else warnings).append(
            f"stable-baselines3 {sb3} installed but the checkpoint was saved by {trained_sb3}")
    if torch_version is None:
        problems.append("torch is not installed in this environment")
    elif trained_torch and trained_torch.split("+")[0] != torch_version.split("+")[0]:
        warnings.append(f"torch {torch_version} installed but the checkpoint was saved with {trained_torch}")

    if obs_dim != OBSERVATION_DIM:
        problems.append(f"observation dimension is {obs_dim}, the Command Centre builds {OBSERVATION_DIM} values")
    try:
        contract = load_contract(file)
        tactics = contract.tactics
        profile, action_profile, encoding = contract.observation_profile, contract.action_profile, contract.previous_action_encoding
        if contract.observation_size != OBSERVATION_DIM:
            problems.append(f"contract expects {contract.observation_size} values, not {OBSERVATION_DIM}")
    except ContractError as error:
        problems.append(f"observation contract: {error}")
    except Exception as error:                      # a missing simulator config etc.: say so, do not crash the AI / MODEL screen
        problems.append(f"observation contract could not be read: {type(error).__name__}: {error}")

    info = ModelInfo(str(file), sha, int(stat.st_size), trained_sb3, sb3, trained_torch, torch_version, obs_dim, action_space, tactics,
                     profile, action_profile, encoding, DETERMINISTIC, False, None, None, not problems, tuple(problems), tuple(warnings))
    with _INFO_LOCK:
        _INFO_CACHE[key] = (sha, info)
    return info


# ----------------------------------------------------------------------------------------------
# The loaded model: one per process


@dataclass
class _Loaded:
    model: Any
    encoding: str
    loaded_at: float
    load_seconds: float


_MODELS: dict[tuple[str, int, str], _Loaded] = {}
_KEY_LOCKS: dict[tuple[str, int, str], threading.Lock] = {}
_MODELS_LOCK = threading.Lock()
LOAD_COUNT = 0                                      # how many real loads this process has done (tests assert it stays at 1)


def _load_model(path: str, device: str):               # the only place a checkpoint is read: patched in tests that count loads
    from autobot.sim import cli
    return cli.load_checkpoint(path, device=device)


def shared_model(path: Path | str = RECOMMENDED_CHECKPOINT, device: str = "cpu") -> tuple[Any, str, ModelInfo]:
    """The process-wide loaded model for this exact file (path + mtime + sha256) and its previous-action encoding.

    The first caller loads it; concurrent and later callers get the same object. Raises on an incompatible or unreadable
    checkpoint (callers that must not raise use ``PolicyEngine``)."""
    global LOAD_COUNT
    info = inspect_model(path)
    if not info.compatible:
        raise ValueError("checkpoint is not compatible: " + "; ".join(info.problems))
    file = Path(info.path)
    key = (info.path, file.stat().st_mtime_ns, info.sha256)
    with _MODELS_LOCK:
        loaded = _MODELS.get(key)
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    if loaded is None:
        with lock:                                       # a second thread waits here, then finds the model already cached
            with _MODELS_LOCK:
                loaded = _MODELS.get(key)
            if loaded is None:
                started = time.perf_counter()
                model, encoding = _load_model(info.path, device)
                if tuple(model.observation_space.shape) != (OBSERVATION_DIM,) or int(model.action_space.n) != len(info.tactics):
                    raise ValueError("the loaded model's spaces differ from its recorded contract")
                loaded = _Loaded(model, encoding, time.time(), time.perf_counter() - started)
                with _MODELS_LOCK:
                    _MODELS[key] = loaded
                    LOAD_COUNT += 1
    done = ModelInfo(**{**info.__dict__, "loaded": True, "loaded_at": loaded.loaded_at, "load_seconds": loaded.load_seconds})
    return loaded.model, loaded.encoding, done


def forget_models() -> None:
    """Drop the cached models (tests, or after replacing the checkpoint file)."""
    global LOAD_COUNT
    with _MODELS_LOCK:
        _MODELS.clear()
        _KEY_LOCKS.clear()
        LOAD_COUNT = 0


# ----------------------------------------------------------------------------------------------
# Decisions


@dataclass(frozen=True)
class PolicyOutput:
    ok: bool
    action: int | None = None
    tactic: str | None = None
    probability: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    error: str = ""
    latency_ms: float | None = None


@dataclass(frozen=True)
class PolicyDecision:
    """What the policy decided at one time. ``tactic`` is None whenever ``blocked_by`` is not empty because of the policy path."""
    tactic: str | None
    action: int | None
    probability: float | None
    probabilities: dict[str, float]
    source: str                                      # LIVE | SIMULATED | REPLAY
    valid_for_control: bool
    blocked_by: tuple[str, ...]
    ages_s: dict[str, float | None]                  # per input group, seconds at decision time (None for constants)
    decided_at: float
    input_time: float | None                         # time of the OLDEST measured input: pass this to command_mapping as decision_time
    model_sha256: str
    observation: tuple[float, ...] | None
    error: str = ""
    latency_ms: float | None = None

    @property
    def blocked(self) -> bool:
        return self.tactic is None

    def proposal(self) -> dict[str, Any] | None:
        """The dictionary the existing shadow UI shows as ``proposed`` (None when there is no tactic)."""
        if self.tactic is None:
            return None
        return {"action": self.action, "tactic": self.tactic, "probability": self.probability,
                "probabilities": dict(self.probabilities), "source": self.source, "valid_for_control": self.valid_for_control}

    def to_report(self) -> dict[str, Any]:
        return {"tactic": self.tactic, "action": self.action, "probability": self.probability, "source": self.source,
                "valid_for_control": self.valid_for_control, "blocked_by": list(self.blocked_by),
                "ages_ms": {k: None if v is None else round(1000 * v, 1) for k, v in self.ages_s.items()},
                "decided_at": self.decided_at, "input_time": self.input_time, "model_sha256": self.model_sha256[:12],
                "error": self.error, "latency_ms": self.latency_ms,
                "output": "none: a decision is data; only command_mapping + the gated sender can turn it into a command"}


@dataclass(frozen=True)
class LoopToken:
    owner: str
    serial: int


class _LoopRegistry:
    """Process-wide: there is ONE inference loop, whatever engine objects exist."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: LoopToken | None = None
        self._serial = 0

    def start(self, owner: str) -> LoopToken:
        with self._lock:
            if self._loop is not None:
                raise DuplicateLoopError(f"an inference loop is already running (owner {self._loop.owner!r}); stop it first")
            self._serial += 1
            self._loop = LoopToken(str(owner), self._serial)
            return self._loop

    def stop(self, token: LoopToken | None = None) -> bool:
        with self._lock:
            if self._loop is None or (token is not None and token != self._loop):
                return False
            self._loop = None
            return True

    @property
    def active(self) -> LoopToken | None:
        with self._lock:
            return self._loop


LOOPS = _LoopRegistry()


class PolicyEngine:
    """One checkpoint, one inference lock, at most one inference loop."""

    def __init__(self, path: Path | str = RECOMMENDED_CHECKPOINT, device: str = "cpu") -> None:
        self.path, self.device = Path(path), device
        self._model = None
        self._info: ModelInfo | None = None
        self._tactics: tuple[str, ...] = ()
        self._predict_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.encoding = ""
        self.load_error = ""
        self.predictions = 0

    # ---- identity and loading ----

    def info(self) -> ModelInfo:
        """Identity/compatibility; reflects ``loaded`` state. Never raises."""
        with self._state_lock:
            if self._info is not None:
                return self._info
        return inspect_model(self.path)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> ModelInfo:
        """Load (once per process) and return the ModelInfo. Never raises: a failure is in ``problems`` and ``load_error``."""
        with self._state_lock:
            if self._model is not None and self._info is not None:
                return self._info
        try:
            model, encoding, info = shared_model(self.path, self.device)
        except Exception as error:
            log.exception("checkpoint load failed")
            self.load_error = f"{type(error).__name__}: {error}"
            base = inspect_model(self.path)
            problems = base.problems if base.problems else (self.load_error,)
            info = ModelInfo(**{**base.__dict__, "compatible": False, "problems": tuple(problems)})
            with self._state_lock:
                self._info = info
            return info
        with self._state_lock:
            self._model, self._info, self._tactics, self.load_error, self.encoding = model, info, info.tactics, "", encoding
        return info

    # ---- inference loop registry (process-wide, see LOOPS) ----

    def start(self, owner: str) -> LoopToken:
        """Register the (single) inference loop. A second ``start`` while one is active raises DuplicateLoopError."""
        return LOOPS.start(owner)

    def stop(self, token: LoopToken | None = None) -> bool:
        """Release the loop. With a token, only that loop is released (a stale token cannot stop a newer loop). Idempotent."""
        return LOOPS.stop(token)

    @property
    def active_loop(self) -> LoopToken | None:
        return LOOPS.active

    @contextlib.contextmanager
    def loop(self, owner: str):
        """``with engine.loop("owner"):`` registers the single inference loop for the block and always releases it."""
        token = LOOPS.start(owner)
        try:
            yield token
        finally:
            LOOPS.stop(token)

    # ---- inference ----

    def predict(self, observation: Any) -> PolicyOutput:
        """Deterministic action for one 19-value observation. Never raises; a failure is ``ok=False`` with the reason."""
        started = time.perf_counter()
        try:
            if self._model is None:
                info = self.load()
                if self._model is None:
                    return PolicyOutput(False, error="model not loaded: " + "; ".join(info.problems))
            vector = np.asarray(observation, dtype=np.float32)
            if vector.shape != (OBSERVATION_DIM,):
                return PolicyOutput(False, error=f"observation has shape {vector.shape}, expected ({OBSERVATION_DIM},)")
            if not np.isfinite(vector).all() or np.abs(vector).max() > 1.0 + 1e-6:
                return PolicyOutput(False, error="observation contains a non-finite value or one outside -1..1")
            import torch
            with self._predict_lock:
                action = int(self._model.predict(vector, deterministic=DETERMINISTIC)[0])
                policy = self._model.policy
                with torch.no_grad():
                    tensor, _ = policy.obs_to_tensor(vector)
                    probabilities = policy.get_distribution(tensor).distribution.probs[0].cpu().numpy()
            if not 0 <= action < len(self._tactics):
                return PolicyOutput(False, error=f"the model returned action {action}, outside 0..{len(self._tactics) - 1}")
            self.predictions += 1
            return PolicyOutput(True, action, self._tactics[action], round(float(probabilities[action]), 4),
                                {name: round(float(p), 4) for name, p in zip(self._tactics, probabilities)},
                                latency_ms=1000 * (time.perf_counter() - started))
        except Exception as error:
            log.exception("policy inference failed")
            return PolicyOutput(False, error=f"inference error: {type(error).__name__}: {error}", latency_ms=1000 * (time.perf_counter() - started))

    def actions(self, observations: Any) -> np.ndarray:
        """Greedy action for each row of a batch (offline analysis). Raises on failure; not used by the live path."""
        if self._model is None:
            info = self.load()
            if self._model is None:
                raise RuntimeError("model not loaded: " + "; ".join(info.problems))
        with self._predict_lock:
            return np.asarray(self._model.predict(np.asarray(observations, np.float32), deterministic=DETERMINISTIC)[0]).reshape(-1)

    def decide(self, result: ObservationResult, now: float | None = None) -> PolicyDecision:
        """An ObservationResult -> a PolicyDecision. Incomplete observation, incompatible model or inference error = blocked."""
        now = result.built_at if now is None else float(now)
        sha = self.info().sha256
        common = dict(source=result.source, ages_s=result.ages_s(), decided_at=now, input_time=result.oldest_input_t(), model_sha256=sha)
        if result.values is None:
            return PolicyDecision(None, None, None, {}, valid_for_control=False, blocked_by=tuple(result.blocked_by), observation=None, **common)
        output = self.predict(result.values)
        observation = tuple(float(v) for v in result.values)
        if not output.ok:
            return PolicyDecision(None, None, None, {}, valid_for_control=False, blocked_by=tuple(result.blocked_by) + (output.error,),
                                  observation=observation, error=output.error, latency_ms=output.latency_ms, **common)
        return PolicyDecision(output.tactic, output.action, output.probability, output.probabilities,
                              valid_for_control=bool(result.valid_for_control), blocked_by=tuple(result.blocked_by),
                              observation=observation, latency_ms=output.latency_ms, **common)


_ENGINES: dict[str, PolicyEngine] = {}
_ENGINE_LOCK = threading.Lock()


def engine_for(path: Path | str = RECOMMENDED_CHECKPOINT, device: str = "cpu") -> PolicyEngine:
    """One PolicyEngine per checkpoint file per process (the model itself is cached separately, see ``shared_model``)."""
    key = f"{Path(path).resolve()}|{device}"
    with _ENGINE_LOCK:
        if key not in _ENGINES:
            _ENGINES[key] = PolicyEngine(path, device)
        return _ENGINES[key]


def get_engine() -> PolicyEngine:
    """The engine for the recommended checkpoint (runs/recovery-challenger-v2-seed1/final_model.zip)."""
    return engine_for(RECOMMENDED_CHECKPOINT)
