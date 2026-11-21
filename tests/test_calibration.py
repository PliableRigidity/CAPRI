"""The calibrate command must finish, or fail fast and clearly. It must never spin on an unstable model."""
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from autobot.sim.cli import CalibrationError, main, run_calibration_window
from autobot.sim.core import AutoBotSim
from autobot.sim.env import AutoBotEnv, tactic_to_drive

FROZEN = Path("runs/ppo-curriculum-overnight/effective_sim.yaml")


def _unstable_config(tmp_path):
    """Same model integrated five times too coarsely: MuJoCo reports an unstable step within 0.1 simulated seconds."""
    config = yaml.safe_load(Path("config/sim.yaml").read_text(encoding="utf-8"))
    config["simulation"]["timestep_s"] = 0.01
    path = tmp_path / "unstable.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


def test_calibrate_completes_cleanly_on_the_current_model(capsys):
    started = time.perf_counter()
    assert main(["calibrate"]) is None
    assert time.perf_counter() - started < 30
    output = capsys.readouterr()
    assert output.err == ""
    assert "Calibration completed: 4 windows, 5610 physics steps, no instability reported." in output.out
    assert output.out.count("simulated=") == 4
    travelled = float(output.out.split("3.00s: simulated=")[1].split("m")[0])
    assert 1.0 < travelled < 2.5


def test_calibration_drives_forward_for_the_exact_window():
    sim = AutoBotSim(seed=7, calibration_track=True); start = sim.truth()["position_m"].copy()
    assert run_calibration_window(sim, tactic_to_drive(1), 1.76) == 880
    assert sim.data.time == pytest.approx(1.76, abs=1e-9)
    assert sim.truth()["position_m"][0] - start[0] > 0.5 and sim.instability() is None


def test_old_wait_for_time_loop_could_not_terminate_on_an_unstable_model(tmp_path):
    """Root cause of the runaway jobs, reproduced with a hard step bound.

    MuJoCo resets the state (time back to zero) whenever a step is unstable, so the old
    `while sim.data.time < window` condition stays true forever. 2,000 physics steps here would be
    20 simulated seconds on a healthy model; on this one simulated time never passes 0.2 s.
    """
    sim = AutoBotSim(_unstable_config(tmp_path), 7, calibration_track=True)
    latest = 0.0; resets = 0; furthest = 0.0
    for _ in range(2000):
        sim.step(tactic_to_drive(1), (0, 0), 1)
        resets += sim.data.time < latest; latest = float(sim.data.time); furthest = max(furthest, latest)
    assert resets > 10 and furthest < 0.2
    flagged = sum(int(sim.data.warning[kind].number) for kind in (
        mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC))
    assert flagged >= 1
    assert "unstable" in sim.instability()


def test_unstable_calibration_fails_fast_with_a_clear_message(tmp_path, capsys):
    config = _unstable_config(tmp_path)
    started = time.perf_counter()
    with pytest.raises(SystemExit) as failure:
        main(["--config", config, "calibrate"])
    assert failure.value.code == 2 and time.perf_counter() - started < 10
    error = capsys.readouterr().err
    assert error.startswith("CALIBRATION FAILED:") and "unstable simulation" in error and "mjWARN_BADQ" in error
    assert "physics steps in a 3.0s window" in error
    # Stopped at the first unstable control chunk, not after thousands of resets.
    sim = AutoBotSim(config, 7, calibration_track=True)
    with pytest.raises(CalibrationError, match=r"mjWARN_BADQ(POS|VEL|ACC) x1 "):
        run_calibration_window(sim, tactic_to_drive(1), 3.0)


def test_calibration_step_time_and_window_limits_are_enforced():
    sim = AutoBotSim(seed=7, calibration_track=True)
    with pytest.raises(CalibrationError, match="step limit: a 3.0s window needs 1500 physics steps"):
        run_calibration_window(sim, tactic_to_drive(1), 3.0, max_steps=1499)
    assert sim.data.time == 0.0  # refused before stepping
    ticks = iter(range(0, 10_000, 100))
    with pytest.raises(CalibrationError, match="time limit: exceeded 60.0s"):
        run_calibration_window(sim, tactic_to_drive(1), 3.0, clock=lambda: next(ticks))
    for window in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(CalibrationError, match="invalid calibration window"):
            run_calibration_window(AutoBotSim(seed=7, calibration_track=True), tactic_to_drive(1), window)
    with pytest.raises(SystemExit):
        main(["calibrate", "--max-steps", "10"])


def test_non_finite_state_and_external_reset_are_reported():
    sim = AutoBotSim(seed=7, calibration_track=True)
    assert sim.instability() is None
    sim.data.qvel[0] = math.nan
    assert sim.instability() == "non-finite qvel"
    # A state that is reset behind the loop's back is caught by the time check even with no warning.
    sim = AutoBotSim(seed=7, calibration_track=True)
    with pytest.raises(CalibrationError, match="the state was reset"):
        run_calibration_window(sim, tactic_to_drive(1), 1.0, after_chunk=lambda current: current.reset())


@pytest.mark.parametrize("config", ["config/sim.yaml", pytest.param(str(FROZEN), marks=pytest.mark.skipif(
    not FROZEN.is_file(), reason="frozen run config not present (runs/ is untracked)"))])
def test_training_environment_reports_no_instability(config):
    """The runaway was a calibration-loop defect; the arena environment itself must stay stable."""
    rng = np.random.default_rng(3)
    for policy in ("random", "advance", "reverse"):
        env = AutoBotEnv(config, "sim_debug", seed=11, action_profile="experimental5_v1")
        env.reset(seed=11)
        for _ in range(600):
            action = int(rng.integers(5)) if policy == "random" else 1 if policy == "advance" else 2
            _, _, terminated, truncated, _ = env.step(action)
            assert env.sim.instability() is None
            if terminated or truncated: break
