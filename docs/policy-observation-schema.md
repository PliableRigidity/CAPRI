# Policy observation schema (verified against the repository)

Checkpoint: `runs/recovery-challenger-v2-seed1/final_model.zip`. Every statement carries a `file:line`; anything that
could not be verified from the repository is marked **UNVERIFIED**. Evidence for everything below is SOFTWARE / SIMULATOR.
No hardware was touched, and no row says a live sensor produces these values.

Machine-readable twin: `python -m control_center.tools.observation_contract` (field list), and
`control_center/backend/policy_engine.py:inspect_model()` (identity / compatibility, `ModelInfo`).

## 1. What this model is (and is not)

| Fact | Value | Source |
|---|---|---|
| Not PPO-trained | "SUPERVISED imitation of a simple reactive baseline (turn toward red, advance when aligned) ... Not trained by PPO. Simulation-only: the observation contains simulator truth." Behaviour cloning, then 4 DAgger rounds; value head untouched. Stored as an SB3 PPO object so `PPO.load` works. | `runs/recovery-challenger-v2-seed1/effective_config.json:3`, `training.json` (`history`, `value_head_untouched`), `autobot/sim/cli.py:129` |
| Teacher | turn toward the opponent, advance when within 12 deg; reads only the pose channels 13-18 | `scripts/reactive_recovery.py:120-133` |
| Architecture | `mlp_64_64` | `effective_config.json:7` |
| Profiles | `observation_profile` = `sim_debug`, `action_profile` = `experimental5_v1`, `previous_action_encoding` = `masked_zero_v1` | Adjacent `effective_config.json:10-12`. This checkpoint predates embedded `autobot_observation` metadata and does **not** carry these values in its zip. |
| Observation space | `Box(-1, 1, (19,), float32)` | zip `data`.`observation_space`; `autobot/sim/env.py:120-121` |
| Action space | `Discrete(5)` | zip `data`.`action_space`; `env.py:118` |
| Normalisation wrapper | none: no `VecNormalize` file in the run folder (only `effective_config.json`, `final_model.zip`, `training.json`); `load_checkpoint` is a bare `PPO.load`; replays feed the raw env vector | `autobot/sim/cli.py:127-135`, `scripts/reactive_recovery.py:376` |
| Deterministic? | Yes. Every evaluation, replay and clone rollout calls `predict(obs, deterministic=True)`; the engine does the same | `autobot/sim/cli.py:313`, `:557`; `scripts/reactive_recovery.py:376`, `:499`; `policy_engine.DETERMINISTIC` |
| Training config | `runs/ppo-curriculum-overnight/effective_sim.yaml` (the file named by the run record); differs from `config/sim.yaml` only in `episode_seconds` 20 vs 120 and YAML formatting | `effective_config.json:14`; diff of the two files |

### Model identity and compatibility with this environment

| | Trained with (zip `system_info.txt`, `_stable_baselines3_version`) | Installed here |
|---|---|---|
| stable-baselines3 | 2.9.2 | 2.9.2 |
| torch | 2.15.0+cpu | 2.15.0+cpu |
| gymnasium / numpy / python | 1.4.0 / 2.4.2 / 3.13.7 | 1.4.0 / 2.4.2 / 3.13.7 |

`sha256 = 92e0c190ae8f6e90145e76474a98dda74df05e8a2dc572596a6f28cf5d699d17`, size 64 426 bytes. Verdict: **compatible in its
original run directory**: it loads via `PPO.load` and has 19 inputs and 5 actions. Its required
`previous_action_encoding=masked_zero_v1` comes only from the adjacent `effective_config.json`; the zip has no embedded
`autobot_observation` record. The checkpoint and sidecar must therefore be copied together. If the zip is separated from
that sidecar, `resolve_previous_action_encoding` falls back to the legacy `onehot_v1` encoding. The shape remains valid,
so a plain `PPO.load` cannot detect the semantic mismatch; behavior may be wrong without a load error. A different SB3
major version is reported as a problem, a different minor/patch as a warning (`policy_engine.inspect_model`).

## 2. The 19 observation values (index order is the vector order)

Built in `autobot/sim/env.py:133-149` (`_obs`), clipped to -1..1 at `:149`. Arena frame: origin at the plain corner, +x and +y along
the two walls, 1.15 m x 1.15 m, pits centred at (0, 1.15) and (1.15, 0) radius 0.25 m, flipper triangle at (1.15, 1.15)
(`effective_sim.yaml:6-22`; walls placed from the origin in `autobot/sim/core.py:104-108`). Position point = centre of the
chassis collision box (the body origin, `core.py:138`; `truth()` `core.py:198-204`). Heading = `atan2(R[1,0], R[0,0])` of the body
(`core.py:202`): direction of the nose (body +x), **counter-clockwise from arena +x seen from above**.

| # | name | unit / frame / scaling | training source | live source (this repo) | constant? |
|---|---|---|---|---|---|
| 0 | gyro_x | rad/s, body frame (x nose, y left, z up), mounting 0 deg; `/ 34.9`, clip | `sim.imu()` `env.py:134-136`, `core.py:224,227,229` | controller IMU telemetry, **not on the baseline firmware** | no |
| 1 | gyro_y | same | same | same | no |
| 2 | gyro_z | same (positive = counter-clockwise) | same | same | no |
| 3 | accel_x | m/s^2, body frame; `/ 39.2`, clip; at rest upright = (0, 0, +9.81) | `core.py:226,228,230`: gravity rotated into the body + N(0, 0.08); **no linear acceleration, vibration or impact** | controller IMU telemetry (see above) | no |
| 4 | accel_y | same | same | same | no |
| 5 | accel_z | same | same | same | no |
| 6-12 | previous_tactic_0..6 | exactly `0.0` (`masked_zero_v1`; 7 slots because `experimental5_v1` reserves 7) | `env.py:137-139`, `env.py:15`, `:26` | none needed | **YES, 0.0** (the only constants) |
| 13 | own_x | `2*x/1.15 - 1`, x metres in the arena frame | `truth(0)` `env.py:146` | camera: ArUco chassis pose, floor map to arena frame | no |
| 14 | own_y | `2*y/1.15 - 1` | `env.py:146` | same | no |
| 15 | sin(own_yaw) | no scaling | `env.py:147` | same (heading) | no |
| 16 | cos(own_yaw) | no scaling | `env.py:147` | same | no |
| 17 | opponent_x | `2*x/1.15 - 1`, opponent chassis centre | `truth(1)` `env.py:148` | ROI tracker position, TRACKED state only | no |
| 18 | opponent_y | `2*y/1.15 - 1` | `env.py:148` | same | no |

Notes
- The policy was never trained to read its previous action (the 7 slots are always 0.0). The "previous action" the Command Centre
  remembers is therefore **not** an observation channel and must not be written into 6-12.
- The IMU channels are simulator values: at rest `(0, 0, +9.81)` plus small noise. A real accelerometer also reports the robot's own
  acceleration. Whether the real IMU's axes match the body frame is **UNVERIFIED** (`config/imu_mounting.yaml:22` `verified: false`).
- Sensitivity measured on random in-arena poses (SIMULATED analysis, 3000 poses, seed 0, using `PolicyEngine.actions`): with the
  rest IMU as reference, simulator-like IMU noise changes 0.1 % of actions, hand-held-jiggle sized noise 6.4 %, impact-sized noise
  36.5 %, an all-zeros IMU 8.6 %, an upside-down gravity 17.8 %. So the IMU group is not ignorable and must never be invented
  (`tests/test_policy_engine.py::test_the_policy_reads_its_imu_channels`).

## 3. Timing

| Fact | Value | Source |
|---|---|---|
| Decision / observation rate in training | 50 Hz: one `step()` = 10 physics steps of 2 ms = 20 ms; the chosen tactic is held for that step | `effective_sim.yaml:2-3`, `core.py:54`, `core.py:182-186` |
| IMU sample rate in the simulator | 100 Hz, last sample held until 10 ms have passed | `effective_sim.yaml:140`, `core.py:216` |
| Latency / dropouts in training | none: pose is exact truth, no camera delay | `env.py:144-148` |
| Live decision rate | camera frame rate; the real latency and the effect of acting on 33-66 ms old poses are **UNVERIFIED** | - |

## 4. Actions

| index | tactic | simulator wheels (left, right), negative = forward | meaning | source |
|---|---|---|---|---|
| 0 | stop | (0, 0) | brake | `env.py:11`, `:52` |
| 1 | advance | (-1, -1) | straight forward | `env.py:52` |
| 2 | reverse | (0.65, 0.65) | straight backward | `env.py:52` |
| 3 | turn_left | (0.7, -0.7) | right wheel forward, left backward: counter-clockwise | `env.py:52` |
| 4 | turn_right | (-0.7, 0.7) | clockwise | `env.py:52` |

Indices 5-6 (arcs) exist only in `arc7_v1` (`env.py:12,16`), not in this checkpoint's 5-action head. The simulator's low-level
controller is `tactic_to_drive` (`env.py:49-55`); the runtime mapping to firmware units is `control_center/backend/command_mapping.py`.

## 5. Live input status, and what is blocked

| Group | What exists | Status |
|---|---|---|
| own pose (13-16) | ArUco chassis tracker -> floor map -> arena frame (`control_center/tools/chassis_tracker.py`, `overlay_core.py`) | tested on synthetic fixtures only; marker offsets / heights / dictionary are placeholders: **UNVERIFIED on the robot**; heading convention (nose, counter-clockwise) depends on the configured marker yaw offsets: **UNVERIFIED** |
| opponent (17-18) | `roi_tracker.RoiOpponentTracker`, TRACKED only (UNCERTAIN / LOST / not selected = group missing, never the last position) | fixture-tested; the tracked box centre on the floor plane vs the chassis-box centre the policy was trained on: **UNVERIFIED** |
| IMU (0-5) | `ELRS-Handoff-Telemetry` sketch defines an `IMU` line; `HubTelemetry` / `LiveTelemetry` convert it | **NOT AVAILABLE on the baseline firmware**: `ELRS-Handoff-Bench.ino:16` and `ELRS-Handoff-WiFi.ino:18` say "No IMU"; the mounting transform is unverified (`config/imu_mounting.yaml:22`). Live inference and transmission are therefore **BLOCKED** (`valid_for_control` is False) |
| previous tactic (6-12) | constant 0.0 | available |

Gating, implemented once in `control_center/backend/observation_builder.py` and shared with `stream_adapter.ShadowAssembler`:

| Rule | Value |
|---|---|
| camera age (own pose, opponent) | <= 0.25 s else STALE |
| IMU age | <= 0.10 s else STALE |
| opponent | state `tracked` only |
| own pose | `measured` or `predicted` (chassis tracker), inside the arena +/- 10 cm |
| IMU | present, finite, mounting verified |
| any group missing or stale | **no vector at all** |
| `source` | LIVE only if every present input is declared LIVE; any SIMULATED -> SIMULATED; else any REPLAY -> REPLAY |
| `valid_for_control` | source LIVE **and** a fresh LIVE IMU **and** a complete vector **and** no caller blockers |

SIMULATED / REPLAY inputs may still produce a tactic (DIAGNOSTIC inference); it is never `valid_for_control`.

## 6. UNVERIFIED (needs a physical test or a measurement)

1. Live IMU: no firmware in the working baseline sends it; axes, units and mounting have never been checked on the robot.
2. Real camera -> arena-frame pose accuracy, heading sign and marker offsets (fixture only).
3. Opponent tracker centre vs chassis centre; behaviour while robots touch.
4. Effect of 15-30 Hz, delayed, noisy live observations on a policy trained on exact 50 Hz truth.
5. That the real wheels turn the way the simulated mixer says (left/right wiring, motor polarity, stiction at small commands).
6. That the policy's tactics are any good outside the simulator (it imitates a simple pursuit teacher; DAgger student win rates
   are in `training.json`, all simulation).
