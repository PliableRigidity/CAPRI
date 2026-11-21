"""control_center/backend/policy_engine.py: identity, compatibility, one load, deterministic inference, one loop.

Evidence class: SOFTWARE. The real checkpoint zip is read; where the real model is loaded it is the process-wide cached copy.
Without stable-baselines3 / torch the model-dependent tests are SKIPPED and the reason is printed in the pytest summary.
"""

import hashlib
import json
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control_center.backend import observation_builder as ob  # noqa: E402
from control_center.backend import policy_engine as pe  # noqa: E402
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT  # noqa: E402

CHECKPOINT = RECOMMENDED_CHECKPOINT
SHA256 = "92e0c190ae8f6e90145e76474a98dda74df05e8a2dc572596a6f28cf5d699d17"      # recorded when this module was written
SIZE = 64426
HAS_CHECKPOINT = CHECKPOINT.is_file()


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


NEEDS_MODEL_REASON = ("SKIPPED: stable-baselines3 / torch are not installed, or runs/recovery-challenger-v2-seed1/final_model.zip is absent; "
                      "the model-dependent policy tests cannot run in this environment")
needs_model = pytest.mark.skipif(not (HAS_CHECKPOINT and _have("stable_baselines3") and _have("torch")), reason=NEEDS_MODEL_REASON)
needs_zip = pytest.mark.skipif(not HAS_CHECKPOINT, reason="SKIPPED: the checkpoint zip is not in runs/")


def observation_for(own, heading_rad, opponent):
    """A 19-value vector from the training layout: IMU at rest (gravity only, as the simulator generates it), 7 zeros, poses."""
    w = 1.15
    return np.array([0, 0, 0, 0, 0, 9.81 / 39.2, 0, 0, 0, 0, 0, 0, 0,
                     2 * own[0] / w - 1, 2 * own[1] / w - 1, np.sin(heading_rad), np.cos(heading_rad),
                     2 * opponent[0] / w - 1, 2 * opponent[1] / w - 1], dtype=np.float32)


@pytest.fixture(autouse=True)
def release_loops():
    yield
    pe.LOOPS.stop()


# ---------------------------------------------------------------- identity and compatibility (zip only)


@needs_zip
def test_model_identity_is_read_from_the_real_zip():
    info = pe.inspect_model(CHECKPOINT)
    assert info.sha256 == hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == SHA256
    assert info.size_bytes == CHECKPOINT.stat().st_size == SIZE
    assert (info.obs_dim, info.action_space, info.tactics) == (19, "Discrete(5)", ("stop", "advance", "reverse", "turn_left", "turn_right"))
    assert (info.observation_profile, info.action_profile, info.previous_action_encoding) == ("sim_debug", "experimental5_v1", "masked_zero_v1")
    with zipfile.ZipFile(CHECKPOINT) as archive:
        assert info.sb3_version_trained == archive.read("_stable_baselines3_version").decode().strip() == "2.9.2"
    assert info.deterministic is True and not info.loaded and info.loaded_at is None
    assert info.to_report()["sha256"] == SHA256 and json.dumps(info.to_report())


@needs_zip
def test_compatibility_with_this_environment_is_reported_honestly():
    info = pe.inspect_model(CHECKPOINT)
    if not (_have("stable_baselines3") and _have("torch")):
        assert info.compatible is False and any("not installed" in p for p in info.problems)
        print("compatibility verdict: NOT COMPATIBLE:", info.problems)
        return
    assert info.sb3_version_installed is not None
    if info.sb3_version_installed == info.sb3_version_trained:
        assert info.compatible is True and info.problems == () and not any("stable-baselines3" in w for w in info.warnings)
    print(f"compatibility: sb3 trained {info.sb3_version_trained} / installed {info.sb3_version_installed}, "
          f"torch trained {info.torch_version_trained} / installed {info.torch_version_installed}, compatible={info.compatible}")


def test_a_missing_or_foreign_file_is_incompatible_not_an_exception(tmp_path):
    missing = pe.inspect_model(tmp_path / "nope.zip")
    assert missing.compatible is False and any("unreadable" in p for p in missing.problems)
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip")
    bad = pe.inspect_model(junk)
    assert bad.compatible is False and any("zip" in p or "readable" in p for p in bad.problems)
    wrong = tmp_path / "wrong.zip"
    with zipfile.ZipFile(wrong, "w") as archive:
        archive.writestr("data", json.dumps({"observation_space": {"_shape": [7]}, "action_space": {"n": 3}}))
        archive.writestr("_stable_baselines3_version", "2.9.2")
    odd = pe.inspect_model(wrong)
    assert odd.compatible is False and odd.obs_dim == 7 and any("observation dimension is 7" in p for p in odd.problems)


@needs_zip
def test_a_different_sb3_major_version_is_a_problem_and_a_minor_one_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "_installed_versions", lambda: ("3.0.0", "2.15.0+cpu"))
    pe._INFO_CACHE.clear()
    try:
        major = pe.inspect_model(CHECKPOINT)
        assert major.compatible is False and any("3.0.0" in p for p in major.problems)
        monkeypatch.setattr(pe, "_installed_versions", lambda: ("2.8.0", "2.15.0+cpu"))
        pe._INFO_CACHE.clear()
        minor = pe.inspect_model(CHECKPOINT)
        assert minor.compatible is True and any("2.8.0" in w for w in minor.warnings)
    finally:
        pe._INFO_CACHE.clear()


# ---------------------------------------------------------------- loaded once


@needs_zip
def test_the_model_is_loaded_once_even_when_many_threads_ask(monkeypatch, tmp_path):
    """The loader itself is counted (and made slow, to force the race) on a COPY of the zip, so the real cached model is untouched.
    The cache key is path + mtime + sha256."""
    import shutil
    copy = tmp_path / "copy.zip"
    shutil.copy(CHECKPOINT, copy)
    shutil.copy(CHECKPOINT.parent / "effective_config.json", tmp_path / "effective_config.json")      # the contract's run record
    calls = []

    class FakeSpace:
        shape = (19,)
        n = 5

    class FakeModel:
        observation_space = FakeSpace()
        action_space = FakeSpace()

    def slow_loader(path, device):
        calls.append(path)
        time.sleep(0.2)
        return FakeModel(), "masked_zero_v1"

    trained_sb3, trained_torch = (lambda i: (i.sb3_version_trained, i.torch_version_trained))(pe.inspect_model(CHECKPOINT))
    monkeypatch.setattr(pe, "_load_model", slow_loader)
    monkeypatch.setattr(pe, "_installed_versions", lambda: (trained_sb3, trained_torch))
    pe._INFO_CACHE.clear()
    saved_models, saved_count = dict(pe._MODELS), pe.LOAD_COUNT
    results = []
    threads = [threading.Thread(target=lambda: results.append(pe.shared_model(copy)[0])) for _ in range(8)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1 and pe.LOAD_COUNT == saved_count + 1
        assert len(results) == 8 and len({id(m) for m in results}) == 1
        pe.shared_model(copy)
        assert len(calls) == 1                                                       # a later call is a cache hit too
    finally:
        pe._MODELS.clear()
        pe._MODELS.update(saved_models)
        pe.LOAD_COUNT = saved_count
        pe._INFO_CACHE.clear()


@needs_model
def test_the_real_model_is_one_object_for_every_engine_and_every_checkpoint_policy_wrapper():
    first = pe.PolicyEngine(CHECKPOINT)
    second = pe.PolicyEngine(CHECKPOINT)
    before = pe.LOAD_COUNT
    info_a, info_b = first.load(), second.load()
    assert first._model is second._model and pe.LOAD_COUNT - before <= 1 and pe.LOAD_COUNT >= 1
    count = pe.LOAD_COUNT
    assert info_a.loaded and info_b.loaded and info_a.loaded_at == info_b.loaded_at and info_a.compatible
    from control_center.tools.observation_contract import load_contract
    from control_center.tools.stream_adapter import CheckpointPolicy
    wrapper = CheckpointPolicy(load_contract(CHECKPOINT))
    wrapper.load()
    assert wrapper.model is first._model and pe.LOAD_COUNT == count                          # the shadow path shares the single copy
    assert pe.engine_for(CHECKPOINT) is pe.engine_for(CHECKPOINT) and pe.get_engine() is pe.engine_for(CHECKPOINT)


# ---------------------------------------------------------------- deterministic inference on a recorded observation


@needs_model
def test_observation_dimension_and_deterministic_repeatability():
    engine = pe.get_engine()
    info = engine.load()
    assert info.obs_dim == 19 and info.action_space == "Discrete(5)" and info.compatible
    recorded = observation_for((0.575, 0.575), 0.3, (0.862, 0.664))                  # opponent dead ahead of a robot at the arena centre
    first = engine.predict(recorded)
    assert first.ok and first.tactic == "advance" and first.action == 1 and first.latency_ms is not None
    for _ in range(25):
        again = engine.predict(recorded.copy())
        assert (again.action, again.tactic, again.probabilities) == (first.action, first.tactic, first.probabilities)
    assert set(first.probabilities) == {"stop", "advance", "reverse", "turn_left", "turn_right"} and abs(sum(first.probabilities.values()) - 1) < 2e-3
    for bad in (np.zeros(18, np.float32), np.zeros(20, np.float32), np.zeros((1, 19), np.float32)):
        output = engine.predict(bad)
        assert output.ok is False and "shape" in output.error


@needs_model
def test_prediction_matches_the_sb3_greedy_choice_and_is_thread_safe():
    engine = pe.get_engine()
    engine.load()
    rng = np.random.default_rng(5)
    rows = np.clip(rng.uniform(-1, 1, size=(40, 19)), -1, 1).astype(np.float32)
    rows[:, 6:13] = 0.0
    greedy = [int(engine._model.predict(row, deterministic=True)[0]) for row in rows]
    assert [engine.predict(row).action for row in rows] == greedy and list(engine.actions(rows)) == greedy
    out: dict[int, list[int]] = {}

    def work(k):
        out[k] = [engine.predict(row).action for row in rows]

    threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(v == greedy for v in out.values())


# ---------------------------------------------------------------- errors become a blocked decision


class ExplodingModel:
    class policy:
        @staticmethod
        def obs_to_tensor(_):
            raise RuntimeError("tensor exploded")

    def predict(self, *_args, **_kwargs):
        raise RuntimeError("predict exploded")


def live_result(origin=ob.SOURCE_LIVE, imu=True):
    now = 50.0
    return ob.build_observation(ob.ObservationInputs(
        origin=origin, own=ob.OwnPoseInput(0.575, 0.575, 0.3, now - 0.02), opponent=ob.OpponentInput(0.862, 0.664, now - 0.02),
        imu=ob.ImuInput((0, 0, 0), (0, 0, 9.81), now - 0.01, None, True) if imu else None), now)


@needs_model
def test_an_inference_error_is_a_blocked_decision_never_an_exception_and_never_an_old_action():
    engine = pe.PolicyEngine(CHECKPOINT)
    engine.load()
    good = engine.decide(live_result())
    assert good.tactic == "advance" and good.valid_for_control is True and good.source == "LIVE" and good.blocked_by == ()
    engine._model = ExplodingModel()                                                  # the next inference fails
    bad = engine.decide(live_result())
    assert bad.tactic is None and bad.action is None and bad.blocked and bad.valid_for_control is False
    assert "inference error" in bad.error and any("predict exploded" in line for line in bad.blocked_by)
    assert bad.proposal() is None and bad.observation is not None                       # the observation is kept for diagnosis; no action is
    assert engine.predict(np.zeros(19, np.float32)).ok is False


@needs_model
def test_a_blocked_observation_gives_a_blocked_decision_without_calling_the_model():
    engine = pe.PolicyEngine(CHECKPOINT)
    engine.load()
    calls = []
    engine.predict = lambda *a, **k: calls.append(a) or pe.PolicyOutput(True, 1, "advance")     # must not be reached
    decision = engine.decide(live_result(imu=False))
    assert decision.tactic is None and decision.valid_for_control is False and calls == []
    assert any("imu" in line for line in decision.blocked_by) and decision.source == "LIVE"
    assert decision.ages_s["imu"] is None and decision.ages_s["own_pose"] == pytest.approx(0.02)


@needs_model
def test_a_decision_from_simulated_or_replayed_inputs_has_a_tactic_but_is_not_valid_for_control():
    engine = pe.PolicyEngine(CHECKPOINT)
    engine.load()
    for origin in (ob.SOURCE_SIMULATED, ob.SOURCE_REPLAY):
        decision = engine.decide(live_result(origin))
        assert decision.tactic == "advance" and decision.source == origin and decision.valid_for_control is False
        assert decision.blocked_by and decision.input_time == pytest.approx(50.0 - 0.02)
    report = engine.decide(live_result(ob.SOURCE_SIMULATED)).to_report()
    assert json.dumps(report) and report["source"] == "SIMULATED" and report["valid_for_control"] is False and report["output"].startswith("none")


def test_an_incompatible_model_blocks_every_decision(tmp_path):
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip")
    engine = pe.PolicyEngine(junk)
    info = engine.load()                                                              # never raises
    assert info.compatible is False and not engine.loaded and engine.load_error
    result = live_result()
    decision = engine.decide(result)
    assert decision.tactic is None and decision.valid_for_control is False and decision.error and "not loaded" in decision.error


# ---------------------------------------------------------------- one inference loop


def test_a_second_inference_loop_is_refused_and_a_stale_token_cannot_stop_a_newer_one():
    engine, other = pe.PolicyEngine(CHECKPOINT), pe.PolicyEngine(CHECKPOINT)           # two engine objects, ONE registry
    token = engine.start("tracking screen")
    assert engine.active_loop == token and other.active_loop == token
    with pytest.raises(pe.DuplicateLoopError, match="tracking screen"):
        engine.start("operate screen")
    with pytest.raises(pe.DuplicateLoopError):
        other.start("wifi sender")
    assert engine.stop(token) is True and engine.stop(token) is False and engine.active_loop is None      # idempotent
    newer = other.start("operate screen")
    assert engine.stop(token) is False and other.active_loop == newer                  # the old token does not stop the new loop
    assert other.stop(newer) and other.start("again").serial > newer.serial


def test_two_threads_racing_to_start_get_exactly_one_loop():
    engine = pe.PolicyEngine(CHECKPOINT)
    winners, losers = [], []

    def start(k):
        try:
            winners.append(engine.start(f"t{k}"))
        except pe.DuplicateLoopError:
            losers.append(k)

    threads = [threading.Thread(target=start, args=(k,)) for k in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1 and len(losers) == 9


# ---------------------------------------------------------------- why the IMU group must never be invented


@needs_model
def test_the_policy_reads_its_imu_channels():
    """SIMULATED analysis behind docs/policy-observation-schema.md: replacing the IMU by zeros (what 'no IMU' would look like if it
    were silently filled in) changes the chosen tactic for a visible share of poses, while simulator-sized noise hardly does."""
    engine = pe.get_engine()
    engine.load()
    rng = np.random.default_rng(0)
    n, w = 2000, 1.15
    rows = np.zeros((n, 19), np.float32)
    own, opp, heading = rng.uniform(0.1, 1.05, (n, 2)), rng.uniform(0.1, 1.05, (n, 2)), rng.uniform(-np.pi, np.pi, n)
    rows[:, 13], rows[:, 14], rows[:, 15], rows[:, 16] = 2 * own[:, 0] / w - 1, 2 * own[:, 1] / w - 1, np.sin(heading), np.cos(heading)
    rows[:, 17], rows[:, 18] = 2 * opp[:, 0] / w - 1, 2 * opp[:, 1] / w - 1
    at_rest = rows.copy()
    at_rest[:, 5] = 9.81 / 39.2
    reference = engine.actions(at_rest)
    noisy = at_rest.copy()
    noisy[:, 0:3] += rng.normal(0, 0.01 / 34.9, (n, 3))
    noisy[:, 3:6] += rng.normal(0, 0.08 / 39.2, (n, 3))
    assert np.mean(engine.actions(np.clip(noisy, -1, 1)) != reference) < 0.01
    assert np.mean(engine.actions(rows) != reference) > 0.03                           # an all-zero IMU is NOT equivalent to a resting one


def test_the_loop_context_manager_always_releases_the_slot():
    engine = pe.PolicyEngine(CHECKPOINT)
    with engine.loop("first") as token:
        assert engine.active_loop == token
        with pytest.raises(pe.DuplicateLoopError):
            with engine.loop("second"):
                pass
        assert engine.active_loop == token                                               # the refused attempt did not release the first
    assert engine.active_loop is None
    with pytest.raises(RuntimeError, match="boom"):
        with engine.loop("crashing"):
            raise RuntimeError("boom")
    assert engine.active_loop is None
