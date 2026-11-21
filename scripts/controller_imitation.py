"""Can a small supervised policy reproduce the turn-then-advance controller from the policy's own observations?

    python scripts/controller_imitation.py

This separates an observation problem from a learning problem. The student is the network PPO trains (SB3
MlpPolicy actor, two tanh layers of 64) and sees exactly what PPO sees: the 19-channel sim_debug observation
with IMU noise and the previous-action channels masked to zero. It is trained by cross-entropy to output the
controller's tactic. If it reproduces the controller and wins the held-out ladder starts, the observations and
the network are sufficient and what PPO lacks is a learning signal; if it cannot, the observation or the
network is the limit.

Data: the controller's own trajectories from the ladder training starts (iteration 0, plain behaviour cloning),
then DAgger: the student drives, the controller labels the states the student reaches, and the student is refit
on everything so far. Hold-out starts are never used for fitting.

The controller reads simulator truth, so the saved students are diagnostics, not policy candidates.
Simulation-only; nothing is deployed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import heading_ladder as experiment  # noqa: E402
import ordinary_start_diagnostic as diagnostic  # noqa: E402

ROOT = experiment.ROOT
ARCHITECTURES = {"mlp_64_64": None, "linear": []}     # None = the PPO default actor; linear = no hidden layer (control)
DATA_SEED = 91017
MIN_GRADIENT_STEPS = 3000
STUDENT_STEP_CAP = 300      # decisions per student-driven data episode (6 s); a stuck student adds no new states after that


class Recorder:
    """Drives an episode with `driver`, records every observation with the controller's choice for that state."""
    def __init__(self, env): self.env = env; self.expert = diagnostic.FlankController(); self.expert.bind(env)

    def episode(self, driver, start, seed, step_cap=None):
        observation, _ = self.env.reset(seed=int(seed), options={"scenario": start})
        observations, labels, taken = [], [], []
        while True:
            label = int(self.expert.predict(observation)[0])
            action = label if driver is None else int(driver.predict(observation, deterministic=True)[0])
            observations.append(observation); labels.append(label); taken.append(action)
            observation, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated or (step_cap and len(labels) >= step_cap): break
        return np.asarray(observations, dtype=np.float32), np.asarray(labels), np.asarray(taken)


def build_student(seed, net_arch):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from autobot.sim import cli
    torch.set_num_threads(1)
    env = make_vec_env(experiment.make_env, n_envs=1, seed=seed)
    kwargs = {} if net_arch is None else {"policy_kwargs": {"net_arch": net_arch}}
    model = PPO("MlpPolicy", env, device="cpu", seed=seed, verbose=0, n_steps=256, batch_size=32, gamma=experiment.GAMMA, **kwargs)
    setattr(model, cli.OBSERVATION_METADATA_ATTRIBUTE,
            cli.observation_metadata(experiment.PROFILE, experiment.ACTION_PROFILE, experiment.ENCODING))
    return model


def logits(model, observations):
    import torch
    return model.policy.get_distribution(torch.as_tensor(observations, dtype=torch.float32)).distribution.logits


def fit(model, observations, labels, epochs, seed, learning_rate=1e-3, batch_size=256):
    """Cross-entropy on the actor only (the value head is untouched and unused)."""
    import torch
    policy = model.policy; policy.set_training_mode(True)
    parameters = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    optimiser = torch.optim.Adam(parameters, lr=learning_rate); generator = torch.Generator().manual_seed(int(seed))
    inputs = torch.as_tensor(observations, dtype=torch.float32); targets = torch.as_tensor(labels, dtype=torch.long)
    # The first dataset is only a few thousand decisions: make sure a fit is at least MIN_GRADIENT_STEPS updates.
    epochs = max(epochs, -(-MIN_GRADIENT_STEPS * batch_size // len(targets)))
    for _ in range(epochs):
        order = torch.randperm(len(targets), generator=generator)
        for begin in range(0, len(order), batch_size):
            index = order[begin:begin + batch_size]
            loss = torch.nn.functional.cross_entropy(logits(model, inputs[index]), targets[index])
            optimiser.zero_grad(); loss.backward(); optimiser.step()
    policy.set_training_mode(False)
    with torch.no_grad():
        output = logits(model, inputs)
        return {"loss": float(torch.nn.functional.cross_entropy(output, targets)), "agreement": float((output.argmax(1) == targets).float().mean())}


def agreement(model, observations, labels, tactics):
    import torch
    with torch.no_grad(): chosen = logits(model, observations).argmax(1).numpy()
    rows = {"decisions": int(len(labels)), "agreement": float((chosen == labels).mean())}
    for index, tactic in enumerate(tactics):
        mask = labels == index
        if mask.any(): rows[f"agreement_when_controller_chose_{tactic}"] = {"decisions": int(mask.sum()), "agreement": float((chosen[mask] == index).mean())}
    return rows


def closed_loop(model, env, starts, seed):
    from autobot.sim.ladder import GreedyActor, run_starts, summarise_episodes
    return summarise_episodes(run_starts(GreedyActor(model), env, starts, [seed]))


def train_student(task):
    name, architecture, seed, iterations, epochs, output_dir = task
    from autobot.sim.env import ACTION_PROFILES
    from autobot.sim.ladder import GreedyActor, summarise_episodes
    tactics = ACTION_PROFILES[experiment.ACTION_PROFILE]["tactics"]
    ladder = experiment.load_ladder(); model = build_student(seed, ARCHITECTURES[architecture])
    env = experiment.make_env(); recorder = Recorder(env)
    train_starts = [start for rung in ladder["rungs"] for start in rung["train"]]
    validation_starts = [start for rung in ladder["rungs"] for start in rung["validation"]]
    turning_holdout = [start for start in ladder["holdout"] if abs(start["heading_offset_deg"]) >= ladder["rungs"][1]["abs_heading_offset_deg"]]
    # Held-out reference decisions: the controller's own trajectories from the hold-out starts (never fitted).
    held = [recorder.episode(None, start, DATA_SEED + 50_000 + index) for index, start in enumerate(ladder["holdout"])]
    held_observations = np.concatenate([item[0] for item in held]); held_labels = np.concatenate([item[1] for item in held])
    first_observations = np.stack([item[0][0] for item, start in zip(held, ladder["holdout"]) if start in turning_holdout])
    first_labels = np.asarray([item[1][0] for item, start in zip(held, ladder["holdout"]) if start in turning_holdout])
    observations = []; labels = []; history = []
    for iteration in range(iterations + 1):
        driver = None if iteration == 0 else GreedyActor(model)
        cap = None if iteration == 0 else STUDENT_STEP_CAP
        disagreements = 0; decisions = 0
        for index, start in enumerate(train_starts):
            new_observations, new_labels, taken = recorder.episode(driver, start, DATA_SEED + 1000 * iteration + index, cap)
            observations.append(new_observations); labels.append(new_labels)
            disagreements += int((taken != new_labels).sum()); decisions += len(new_labels)
        all_observations = np.concatenate(observations); all_labels = np.concatenate(labels)
        fitted = fit(model, all_observations, all_labels, epochs, seed + iteration)
        row = {"iteration": iteration, "data": "controller trajectories" if iteration == 0 else "student-visited states, controller labels",
               "new_decisions": decisions, "student_disagreed_while_driving": None if iteration == 0 else disagreements / max(decisions, 1),
               "dataset_decisions": int(len(all_labels)), "label_counts": {tactics[k]: int(v) for k, v in sorted(Counter(all_labels.tolist()).items())},
               "fit": fitted, "heldout_controller_trajectories": agreement(model, held_observations, held_labels, tactics),
               "heldout_first_decision_when_a_turn_is_needed": agreement(model, first_observations, first_labels, tactics)}
        if iteration in (0, iterations):
            row["closed_loop_validation_starts"] = closed_loop(model, env, validation_starts, experiment.FINAL_SEEDS[0])
            row["closed_loop_holdout_needing_a_turn"] = closed_loop(model, env, turning_holdout, experiment.FINAL_SEEDS[0])
        history.append(row)
        if iteration == 0 and output_dir: _save(model, Path(output_dir) / "behaviour_cloning_only", name, architecture, seed, 0, epochs)
    path = None
    if output_dir: path = _save(model, Path(output_dir) / "final_model", name, architecture, seed, iterations, epochs)
    final = None
    if path:
        _, final = experiment._evaluate_policy((name, {"kind": "checkpoint", "path": str(path)}, ladder, experiment.transfer_scenarios()))
        groups = {}
        for episode in final["transfer"]: groups.setdefault(episode["group"], []).append(episode)
        first_gated = ladder["rungs"][1]["abs_heading_offset_deg"]
        final = {"path": final["path"], "holdout_by_abs_offset_deg": experiment.by_offset(final["holdout"]),
                 "holdout_needing_a_turn": summarise_episodes([e for e in final["holdout"] if abs(e["heading_offset_deg"]) >= first_gated]),
                 "transfer": {group: summarise_episodes(rows) for group, rows in groups.items()}}
    return name, {"architecture": architecture, "seed": seed, "dagger_iterations": iterations, "epochs_per_fit": epochs,
                  "actor_parameters": int(sum(p.numel() for p in list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters()))),
                  "history": history, "final_closed_loop": final}


def _save(model, path, name, architecture, seed, iterations, epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.parent / "effective_config.json"
    if not sidecar.exists():
        sidecar.write_text(json.dumps({
            "trainer": "scripts/controller_imitation.py",
            "training": "SUPERVISED imitation (DAgger) of the privileged turn-then-advance controller. Not trained by PPO; a diagnostic, not a policy candidate.",
            "student": name, "architecture": architecture, "seed": seed, "dagger_iterations": iterations, "epochs_per_fit": epochs,
            "profile": experiment.PROFILE, "action_profile": experiment.ACTION_PROFILE, "previous_action_encoding": experiment.ENCODING,
            "reward_profile": "not used", "config_path": str(experiment.CONFIG.resolve()), "ladder": experiment.relative(experiment.LADDER_PATH)}, indent=1),
            encoding="utf-8")
    model.save(path)
    return path.with_suffix(".zip")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); parser.add_argument("--architectures", nargs="+", choices=list(ARCHITECTURES), default=list(ARCHITECTURES))
    parser.add_argument("--iterations", type=int, default=8, help="DAgger iterations after plain behaviour cloning")
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--run-prefix", default="runs/controller-imitation"); parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "artifacts/controller-imitation.json"))
    args = parser.parse_args()
    tasks = []
    for architecture in args.architectures:
        for seed in args.seeds:
            name = f"{architecture}_seed{seed}"; directory = ROOT / f"{args.run_prefix}-{architecture.replace('_', '-')}-seed{seed}"
            if not args.no_save and directory.exists() and any(directory.iterdir()): raise SystemExit(f"Refusing to overwrite {directory}")
            tasks.append((name, architecture, seed, args.iterations, args.epochs, None if args.no_save else str(directory)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool: students = dict(pool.map(train_student, tasks))
    output = {"purpose": "Can a small supervised policy reproduce the turn-then-advance controller's choices from the policy's own observations?",
              "observation": "19-channel sim_debug observation with IMU noise, previous-action channels masked to zero (what PPO sees)",
              "architectures": {"mlp_64_64": "SB3 MlpPolicy actor, the network PPO trains", "linear": "no hidden layer (control: is a non-linear map needed?)"},
              "ladder": experiment.relative(experiment.LADDER_PATH),
              "definitions": {"heldout_controller_trajectories": "controller-driven episodes from the hold-out starts; share of decisions where the student picks the controller's tactic",
                              "heldout_first_decision_when_a_turn_is_needed": "first decision on hold-out starts whose offset a plain advance does not win",
                              "closed_loop": "the student drives; outcomes on starts it was never fitted on",
                              "iteration_0": "plain behaviour cloning from controller trajectories; later iterations are DAgger"},
              "students": students}
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(args.output)
    for name, student in students.items():
        for row in (student["history"][0], student["history"][-1]):
            held = row["heldout_controller_trajectories"]; first = row["heldout_first_decision_when_a_turn_is_needed"]
            loop = row.get("closed_loop_holdout_needing_a_turn") or {}
            print(f"{name:18s} iter {row['iteration']}  held-out agreement {held['agreement']:.3f}  first decision {first['agreement']:.3f}  "
                  f"closed-loop turn hold-out {loop.get('wins')}/{loop.get('episodes')} ({loop.get('own_falls')} falls)")


if __name__ == "__main__":
    main()
