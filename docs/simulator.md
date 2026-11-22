# Capri simulator and learning experiments

How to run the MuJoCo simulator, the training and evaluation commands, and the simulation-only experiments.
This text was the simulator section of the repository README until 22 November 2026 and is otherwise unchanged.

**Everything here is simulation only.** No command in this document touches the robot. The story of what these
experiments found is [sections 5 and 6 of the case study](case-study/README.md#5-teaching-blue-to-push-what-failed-and-what-worked).

Things to know before reading:

- The Python package is `autobot` and its classes are `AutoBotSim` and `AutoBotEnv`. The names predate the robot's
  name, Capri, and are kept because checkpoints, scripts and tests depend on them.
- The policy recommended for live simulator play is a behaviour-cloned copy of a scripted controller, not a
  PPO-trained policy ([schema and audit](policy-observation-schema.md)). The `train` command below runs PPO, which did
  not learn the pushing task in these experiments.
- `runs/` and `artifacts/` are git-ignored. Commands that name files under them need those files to exist locally.
- The frozen 20 s evaluation settings live in `runs/ppo-curriculum-overnight/effective_sim.yaml`; `config/sim.yaml`
  has a 120 s horizon for manual play.

## Preview, calibration and checks

The standalone simulator is intentionally separate from live motor control and uses MuJoCo as
its only physics engine. Open the clean top-down view (which reads MuJoCo state directly) with:

```powershell
python -m autobot.sim.cli preview --opponent scripted --opponent-chassis last_year
python -m autobot.sim.cli preview3d --opponent-chassis last_year --view overhead
python -m autobot.sim.cli preview3d --opponent-chassis last_year --view side
```

Keys: WASD/arrows drive, Space pauses, R resets. The shared configuration is
`config/sim.yaml`. Run physical-versus-sim timing windows and all headless checks with:

```powershell
python -m autobot.sim.cli calibrate
python -m autobot.sim.cli check
```

The 2D preview starts paused so both spawn poses can be inspected. Space starts the round.
Scripted play is the default; use either mode explicitly with:

```powershell
python -m autobot.sim.cli preview --opponent scripted
python -m autobot.sim.cli preview --opponent static
```

`assets/BattleBot v3.2.stl` is configured as the current honeycomb-front robot (Capri V1) and
`assets/Opp_Chassis.stl` as `last_year`. Both exports are millimetres with +z up, -y nose, and
include wheels; they are scaled uniformly by 0.001 and rotated into MuJoCo +x-forward. Their
measured overall STL bounds are respectively 162.0 x 135.3424 x 33.4868 mm and
110.0 x 125.1623 x 34.0601 mm. Detailed meshes are visual-only and contribute no collision mass.
Transparent rear-wheel collision bodies and simple chassis boxes remain the physics model. Review
them interactively with `--collision-overlay`, or save alignment images with:

```powershell
python -m autobot.sim.cli snapshot3d --view overhead --output artifacts/overhead.png
python -m autobot.sim.cli snapshot3d --view side --collision-overlay --output artifacts/side-overlay.png
```

Capri V1's 33.5 mm wheel diameter and approximately 132 mm spacing are physical measurements. Its
26 mm rear-axle offset is inferred from the STL. The old bot's 34 mm diameter, 90 mm spacing, and
60 mm rear offset are STL-derived placeholders pending confirmation. All model choices, transforms,
outlines, axle positions, collision footprints, masses, and drive values are in `config/sim.yaml`.

The scripted opponent uses simulator truth only inside its baseline controller and sends ordinary
wheel commands through the shared MuJoCo drive interface. Its manual-play drive authority is lower
than the player's full command, and it stops applying thrust at the configured contact distance so it
cannot pin the manually driven bot indefinitely. It does not add truth to PPO observations.
Four solid MuJoCo walls have their inner faces on the nominal 1.150 m boundary; wall contact is not
a loss. The floor is absent inside both R250 openings. A round ends as `Pit fall` only when MuJoCo
reports a chassis centre below the configured fall height, so partial support over a visual,
non-colliding red rim is allowed. The static triangular flipper is neither an exit nor a pit. Passing
through an outer wall is reported as a simulation bug and truncates training rather than awarding a loss.

The PPO action is a tactic (`stop`, `advance`, `reverse`, `turn_left`, or `turn_right`), mapped
by a deterministic low-level controller; PPO never commands wheel velocities directly. Start a
short smoke test or the configured full run with:

```powershell
python -m autobot.sim.cli train --timesteps 512 --output runs/smoke --checkpoint-freq 256 --eval-freq 256 --eval-episodes 2
python -m autobot.sim.cli train --timesteps 100000 --output runs/ppo
python -m autobot.sim.cli train --timesteps 100000 --output runs/ppo-resumed --resume runs/ppo/final_model.zip
python -m autobot.sim.cli evaluate --model runs/ppo/final_model.zip --episodes 10
python -m autobot.sim.cli evaluate --episodes 10
python -m autobot.sim.cli replay --model runs/ppo/final_model.zip
```

`hardware_plausible` observations contain only configurable simulated gyro/accelerometer data
and the previous tactic. They contain no absolute pose, heading, opponent location, or camera.
The optional `sim_debug` profile adds privileged truth and every resulting checkpoint is labelled
simulation-only. The current reward, pits, static flipper, spawn headings, chassis box, mass,
friction, and drive response are documented preview assumptions rather than official rules or a
calibrated/inverted-validated chassis. The STL visuals are separate from collision geometry and
can be replaced or disabled without changing policy observations or the low-level controller.

## Previous-action observation encoding (versioned)

The previous-tactic channels have a versioned encoding, chosen at training time with
`--previous-action-encoding`:

- `onehot_v1` (default, and what every checkpoint before versioning used): one-hot of the last tactic.
- `masked_zero_v1`: the same channels held at 0.0 at reset and on every step, in training and evaluation.

Both encodings have the same observation shape and layout, so a shape check cannot tell them apart.
The setting is therefore written to the run's `effective_config.json` and embedded in every checkpoint
zip. `replay`, `play`, `evaluate`, `benchmark`, and `train --resume` restore it from the checkpoint and
refuse an explicitly conflicting flag; checkpoints with no record are treated as `onehot_v1`.

```powershell
python -m autobot.sim.cli train --timesteps 100000 --output runs/ppo-masked --previous-action-encoding masked_zero_v1
python -m autobot.sim.cli replay --model runs/ppo-masked/final_model.zip
```

## PPO discount and thread count

`train --gamma` sets the PPO discount (default 0.99, the value every earlier run used; must be in (0, 1]).
The value PPO actually uses is read back from the model and written to `effective_config.json`
(`gamma`, `ppo_hyperparameters`) and to a `run_start` line in `metrics.jsonl`; Stable-Baselines3 also stores
it in every checkpoint. `train --resume` keeps the checkpoint's discount and refuses a different `--gamma`.
`train --torch-threads N` sets an explicit Torch thread count. Use 1 when running several jobs at once:
the default of one thread per core makes concurrent jobs starve each other. Thread count changes
floating-point results, so matched arms must share it.

`metrics.jsonl` also records each `curriculum_stage_change` (from the stage of the episodes actually
sampled) and, per training episode, the scenario name, length, and first-contact step.

```powershell
python -m autobot.sim.cli train --timesteps 1000000 --output runs/ppo-gamma --gamma 0.999 --torch-threads 1
```

## Calibration command limits

`calibrate` runs each window for a fixed number of physics steps and exits non-zero with
`CALIBRATION FAILED: ...` if MuJoCo reports an unstable step, the state becomes non-finite, simulated time
stops advancing, or a step or wall-clock limit is hit (`--max-steps`, `--max-wall-seconds`, per window).
Straight-line windows use the advance tactic.

## Heading ladder (simulation-only experiment)

`scripts/heading_ladder.py` asks whether PPO can learn turn-then-push when every training episode is a
misaligned near-pit start. Red sits 0.34-0.58 m from a pit facing blue; blue is 0.18 m behind it with its
heading rotated off the pit line. A plain advance wins up to 12 degrees and nothing from 14 degrees; a
hand-written turn-then-advance controller wins both. Rungs are heading offsets (10 degrees ungated warm-up,
then 16, 24, 34, 45, 60, 90). `make` refuses to write a ladder whose gated rungs do not separate
Always-Advance from the controller. `train` moves to the next rung only after the greedy policy beats
Always-Advance on that rung's separate validation starts in two consecutive validations, and stops when a
rung is not cleared within its step budget. `--pits` builds a single-pit ladder. `evaluate` scores hold-out
starts (unused pit/distance/angle combinations and in-between offsets) and, separately, ordinary starts.

```powershell
python scripts/heading_ladder.py make
python scripts/heading_ladder.py train --seed 23 --output runs/heading-ladder-seed23
python scripts/heading_ladder.py evaluate --runs runs/heading-ladder-seed23
```

`scripts/controller_imitation.py` fits the PPO actor network by supervised learning to the controller's
choices from the same observations, and `scripts/pit_symmetry_probe.py` compares a checkpoint's tactic
preferences at mirrored states of the two pits. Both are diagnostics; nothing they produce is a policy
candidate.

## Clone fine-tuning pilot (simulation-only experiment)

`scripts/clone_finetune.py` asks whether PPO keeps, loses or extends turn-then-push when its actor starts
as the behaviour-cloned copy of the controller. Arms share seeds, start order and step budget (348,160
steps): the frozen clone, PPO from the clone after a critic-only warm-up, PPO from scratch, and PPO from the
clone with an untrained critic. The training mix (`config/finetune_mix_v1.json`) is half ladder training
starts that need a turn and half new ordinary starts; `make` refuses any training start that repeats a
held-out case. `verify` must pass before training: it checks that the imported actor gives the clone's
action probabilities on saved observations and reproduces its saved closed-loop episodes. Checkpoints are
indexed by actor updates; `monitor` scores them on held-out starts and `evaluate` writes the comparison and
applies a recommendation rule fixed in the script before any run.

```powershell
python scripts/clone_finetune.py make
python scripts/clone_finetune.py verify
python scripts/clone_finetune.py train --arm finetune --seed 17
python scripts/clone_finetune.py monitor
python scripts/clone_finetune.py evaluate
```

## Combined behaviour cloning (simulation-only experiment)

`scripts/combined_clone.py` asks whether one supervised 64-64 clone can hold both the turning drill and
ordinary starts. The teacher is the scripted turn-then-advance controller; no PPO is run. `collect` records
the controller on the original drill starts (it must reproduce the original 4,909 decisions) and on 120 new
ordinary starts. `train` fits three recipes on three seeds: plain concatenation, minibatches balanced
between drill and ordinary decisions, and balanced plus DAgger. `validate` picks the recipe on validation
starts only, and `evaluate` compares the clones with the drill-only clone, Always-Advance and the controller
on the held-out starts and applies a keep-as-demo rule fixed in the script. The clones read simulator truth
through `sim_debug`, so they are simulator demos, not deployable policies.

```powershell
python scripts/combined_clone.py make
python scripts/combined_clone.py collect
python scripts/combined_clone.py train
python scripts/combined_clone.py validate
python scripts/combined_clone.py evaluate
```

## Live play against a frozen checkpoint, and harder scripted reds (simulation-only)

`play` lets a person drive red (W/S throttle, A/D steer, SPACE start/pause, R reset) against a frozen blue
checkpoint. The 2D viewer runs at real-time speed whatever the render rate: the control step stays 20 ms and
the number of steps per frame follows the wall clock, with paused time not counted. Every session writes its
own log, named from `--input-log` plus the session's start time, with one JSON line per control step stamped
with simulation time and one `reset` line per press of R:

```powershell
python -m autobot.sim.cli --config runs/ppo-curriculum-overnight/effective_sim.yaml --seed 10007 play --model runs/combined-clone-balanced-dagger-seed2/final_model.zip --profile sim_debug --action-profile experimental5_v1 --eval-scenarios runs/ppo-curriculum-overnight/evaluation_scenarios.json --scenario-index 2 --input-log runs/overnight-2026-11-17/live/reference-normal-spawn.jsonl
```

`scripts/verify_play_mode.py` runs that viewer loop headlessly with synthetic key input and checks the
controls, neutral on key release and focus loss, pause, reset, round end, the log, the pace at several render
rates and that recorded replays still reproduce. `--measure-real-time` times a full 20 s timeout with the
real wall clock. None of it measures play against a person.

`autobot/sim/opponents.py` adds seeded red driving styles without touching the original opponent modes:
`persistent_follower`, `curved_approach`, `stop_and_go`, `reactive_changes`, and `pit_pusher` (kept out of
training and selection). `StyledOpponentEnv` hands their commands to the unchanged environment through its
existing override, and can sample a style at reset. `scripts/opponent_pool.py` fixes training, validation and
final-test cases in `config/opponent_pool_v1.json`, benchmarks policies on them by style and start family,
checks from red's applied commands that the styles really drive differently, and counts red self-falls
separately from wins blue caused. `export-replay` writes one case's red commands as a `play
--red-input-replay` log.

```powershell
python scripts/opponent_pool.py benchmark --split validation
python scripts/opponent_pool.py styles
python scripts/opponent_pool.py decide
python scripts/opponent_pool.py benchmark --split test
```
