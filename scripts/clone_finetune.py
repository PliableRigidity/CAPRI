"""Does PPO keep, lose or extend turn-then-push when it starts from the behaviour-cloned actor? Simulation-only pilot.

    python scripts/clone_finetune.py make        # training mix + a new ordinary confirmation set (written once)
    python scripts/clone_finetune.py verify      # importing the clone preserves its probabilities and closed-loop baseline
    python scripts/clone_finetune.py train --arm finetune --seed 17
    python scripts/clone_finetune.py monitor     # periodic checkpoints on held-out starts (cached per checkpoint)
    python scripts/clone_finetune.py evaluate    # final matched comparison and the recommendation rule

Fixed for every run, as in scripts/heading_ladder.py: frozen 20 s config, sim_debug observations, experimental5_v1,
masked_zero_v1 previous action, outcome_only reward, scripted opponent, gamma 0.99, the PPO settings of
`autobot.sim.cli train`. The environment class is the unchanged AutoBotEnv; the mix is an ordinary one-stage
training curriculum.

Arms (same seeds, same episode-by-episode start sequence, same number of environment steps):
    frozen clone          the plain behaviour-cloned 64-64 actor, never updated (control)
    finetune              actor copied from the clone; critic warmed up with the actor frozen; then PPO
    scratch               PPO from its usual random initialisation
    finetune_cold_critic  extra control: actor copied from the clone, critic left at its random initialisation

Critic and optimiser. Every training arm builds PPO exactly as the scratch arm does (same seed, so the same
random critic and the same untouched Adam), then the two fine-tuning arms overwrite the actor with the clone's
weights. Nothing else comes from the clone file: its value head was never trained and its optimiser never used.
`finetune` then spends WARMUP_STEPS collecting rollouts with the actor frozen (requires_grad off), so only the
critic learns the clone's values, and re-creates Adam before the first actor update so no moments carry over.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import heading_ladder as experiment  # noqa: E402
import ordinary_start_diagnostic as diagnostic  # noqa: E402

ROOT = experiment.ROOT
CONFIG = experiment.CONFIG
PROFILE, ACTION_PROFILE, ENCODING, GAMMA = experiment.PROFILE, experiment.ACTION_PROFILE, experiment.ENCODING, experiment.GAMMA
FINAL_SEEDS = experiment.FINAL_SEEDS
MIX_PATH = ROOT / "config/finetune_mix_v1.json"
CONFIRMATION_PATH = ROOT / "config/finetune_confirmation_ordinary_v1.json"
OBSERVATIONS_PATH = ROOT / "artifacts/clone-finetune-import-observations.npz"
IMPORT_CHECK_PATH = ROOT / "artifacts/clone-finetune-import-check.json"
LADDER_RESULTS_PATH = ROOT / "artifacts/heading-ladder-results.json"
MIX_SEED = 11; CONFIRMATION_SEED = 93017
ORDINARY_PER_ARCHETYPE = 10; CONFIRMATION_PER_ARCHETYPE = 12
JITTER_M = 0.06; JITTER_RAD = 0.25              # the recipe of the earlier fresh normal starts
SAME_CASE = {"position_m": 0.02, "yaw_rad": 0.05}
PAIRS = {17: 0, 23: 1, 31: 2}                    # PPO seed -> seed of the behaviour-cloned actor it starts from
ARMS = ("finetune", "finetune_cold_critic", "scratch")
ROLLOUT = 256
WARMUP_STEPS = 160 * ROLLOUT                     # critic-only phase of the `finetune` arm
PPO_STEPS = 1200 * ROLLOUT                       # steps with actor updates in the `finetune` arm
TOTAL_STEPS = WARMUP_STEPS + PPO_STEPS           # environment steps of every training arm
ACTOR_UPDATES = (0, 1, 4, 10, 20, 40, 80, 160, 320, 640, 960, 1200)     # checkpoints, counted in PPO actor updates
SCRATCH_MONITOR_UPDATES = (0, 80, 320, 640, 1200)                        # scratch is all 20 s timeouts: monitor fewer
PIT_NAMES = ("top_left", "bottom_right")
# Decided before any training run. The candidate is the final checkpoint of a `finetune` run; periodic
# evaluations are for diagnosis only, so the held-out starts are never used to pick a checkpoint.
RULE = {"candidate": "final checkpoint of a finetune run",
        "keeps_turning": "wins at least 90% of the frozen clone's wins on the held-out turn-needed starts, overall and at each pit",
        "improves_ordinary": "on the new ordinary confirmation starts, wins exceed both the frozen clone's and Always-Advance's "
                             "by at least 10% of episodes, with no more own falls than the frozen clone; and wins also exceed "
                             "both on the earlier fresh normal starts",
        "turn_share": 0.9, "ordinary_margin": 0.10}


def clone_path(clone_seed):
    return ROOT / f"runs/controller-imitation-mlp-64-64-seed{clone_seed}/behaviour_cloning_only.zip"


def run_directory(arm, seed):
    return ROOT / f"runs/clone-pilot-{arm.replace('_', '-')}-seed{seed}"


def path_label(path):
    """Repository-relative when the file is inside the repository, absolute otherwise."""
    path = Path(path).resolve()
    return experiment.relative(path) if ROOT in path.parents else str(path)


def checkpoint_steps(arm):
    """{environment step: actor updates done}. Checkpoints are saved before the rollout that starts at that step."""
    offset = WARMUP_STEPS if arm == "finetune" else 0
    steps = {offset + ROLLOUT * updates: updates for updates in ACTOR_UPDATES}
    steps.setdefault(0, 0); steps[TOTAL_STEPS] = (TOTAL_STEPS - offset) // ROLLOUT
    return dict(sorted(steps.items()))


# ---------------------------------------------------------------- scenarios

def same_case(a, b):
    """Two starts are the same case when both robots are within 2 cm and 0.05 rad of each other's poses."""
    return all(abs(a[bot][0] - b[bot][0]) <= SAME_CASE["position_m"] and abs(a[bot][1] - b[bot][1]) <= SAME_CASE["position_m"]
               and abs(wrap_angle(a[bot][2] - b[bot][2])) <= SAME_CASE["yaw_rad"] for bot in ("player", "opponent"))


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def placeable(start, arena):
    """Both robots clear of walls, pits and the flipper corner, and clear of each other."""
    pits = np.asarray(arena["pit_centres_m"], dtype=float); points = [np.asarray(start[bot][:2], dtype=float) for bot in ("player", "opponent")]
    for point in points:
        if not (0.10 <= point[0] <= arena["width_m"] - 0.10 and 0.10 <= point[1] <= arena["height_m"] - 0.10): return False
        if np.min(np.linalg.norm(pits - point, axis=1)) < arena["pit_radius_m"] + 0.10: return False
        if point[0] + point[1] > arena["width_m"] + arena["flipper_vertices_m"][0][0] - 0.12: return False
    return bool(np.linalg.norm(points[0] - points[1]) >= 0.30)


def jittered(archetypes, per_archetype, rng, prefix, group, forbidden, arena):
    """Seeded offsets of each archetype; rejects unplaceable starts and any that repeat a forbidden case."""
    made = []
    for item in archetypes:
        count = 0; attempts = 0
        while count < per_archetype:
            attempts += 1
            if attempts > 2000: raise RuntimeError(f"Could not place {per_archetype} starts around {item['name']}")
            pose = lambda values: [round(float(values[0] + rng.uniform(-JITTER_M, JITTER_M)), 4),
                                   round(float(values[1] + rng.uniform(-JITTER_M, JITTER_M)), 4),
                                   round(float(values[2] + rng.uniform(-JITTER_RAD, JITTER_RAD)), 4)]
            start = {"name": f"{prefix}_{item['name']}_{count}", "group": group, "derived_from": item["name"],
                     "player": pose(item["player"]), "opponent": pose(item["opponent"])}
            if not placeable(start, arena) or any(same_case(start, other) for other in list(forbidden) + made): continue
            made.append(start); count += 1
    return made


def heldout_starts(ladder):
    """Everything that must never be trained on: ladder validation and hold-out starts, and the ordinary transfer starts."""
    return ([start for rung in ladder["rungs"] for start in rung["validation"]] + list(ladder["holdout"])
            + experiment.transfer_scenarios())


def build_mix(ladder, arena):
    from autobot.sim.scenarios import load_scenario_file, validate_curriculum
    frozen = load_scenario_file(diagnostic.FROZEN / "training_curriculum.json")
    normal = [item for item in diagnostic.base.scenarios() if item["group"] == "normal"]
    held = heldout_starts(ladder)
    turn = [dict(start, group="turn_needed") for rung in ladder["rungs"] if not rung["warmup"] for start in rung["train"]]
    ordinary = jittered(frozen["groups"]["ordinary"], ORDINARY_PER_ARCHETYPE, np.random.default_rng(MIX_SEED),
                        "mix", "ordinary", held, arena)
    confirmation = jittered(normal, CONFIRMATION_PER_ARCHETYPE, np.random.default_rng(CONFIRMATION_SEED),
                            "pilot_confirm", "ordinary_confirmation", held + ordinary, arena)
    mix = {"schema": "finetune_mix_v1", "seed": MIX_SEED,
           "description": "Clone fine-tuning pilot: half the episodes are ladder training starts that need a turn "
                          "(16-90 degrees, no warm-up rung), half are new ordinary starts. One stage, no gating.",
           "stages": [{"name": "mix", "until_episode": None, "group_weights": {"turn_needed": 0.5, "ordinary": 0.5}}],
           "groups": {"turn_needed": turn, "ordinary": ordinary},
           "provenance": {"turn_needed": "training starts of the gated rungs of config/heading_ladder_v1.json",
                          "ordinary": f"{ORDINARY_PER_ARCHETYPE} seeded offsets (+-{JITTER_M} m, +-{JITTER_RAD} rad) of each ordinary "
                                      "start of the frozen training curriculum; none repeats a held-out case",
                          "same_case_tolerance": SAME_CASE}}
    validate_curriculum(mix); check_disjoint(mix, held + confirmation)
    return mix, confirmation


def check_disjoint(mix, held):
    """No training start of the mix repeats a held-out case (raises otherwise)."""
    clashes = [(start["name"], other["name"]) for group in mix["groups"].values() for start in group for other in held if same_case(start, other)]
    if clashes: raise ValueError(f"Training starts repeat held-out cases: {clashes[:3]}")


def load_mix():
    from autobot.sim.scenarios import validate_curriculum
    mix = json.loads(MIX_PATH.read_text(encoding="utf-8")); validate_curriculum(mix)
    confirmation = json.loads(CONFIRMATION_PATH.read_text(encoding="utf-8"))
    check_disjoint(mix, heldout_starts(experiment.load_ladder()) + confirmation)
    return mix, confirmation


def make(args):
    from autobot.sim.core import load_config
    if MIX_PATH.exists() or CONFIRMATION_PATH.exists(): raise SystemExit("Mix or confirmation file already exists; not overwriting.")
    mix, confirmation = build_mix(experiment.load_ladder(), load_config(str(CONFIG))["arena"])
    MIX_PATH.write_text(json.dumps(mix, indent=1), encoding="utf-8")
    CONFIRMATION_PATH.write_text(json.dumps(confirmation, indent=1), encoding="utf-8")
    print(MIX_PATH, {name: len(group) for name, group in mix["groups"].items()}); print(CONFIRMATION_PATH, len(confirmation))


# ---------------------------------------------------------------- model construction

def training_env(mix, seed):
    """The unchanged environment with the mix as its training curriculum; the start order depends on (mix, seed) only."""
    from stable_baselines3.common.env_util import make_vec_env
    from autobot.sim.env import AutoBotEnv
    curriculum = dict(mix, seed=int(mix["seed"]) + 1000 * int(seed))
    return make_vec_env(lambda: AutoBotEnv(str(CONFIG), PROFILE, opponent_model="last_year", opponent_mode="scripted",
                                           training_curriculum=curriculum, reward_profile="outcome_only",
                                           action_profile=ACTION_PROFILE, previous_action_encoding=ENCODING), n_envs=1, seed=seed)


def actor_modules(policy): return [policy.mlp_extractor.policy_net, policy.action_net]
def critic_modules(policy): return [policy.mlp_extractor.value_net, policy.value_net]
def parameters_of(modules): return [parameter for module in modules for parameter in module.parameters()]


def flat(modules):
    import torch
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters_of(modules)]).numpy().copy()


def import_actor(model, clone):
    """Overwrite the actor (both hidden layers and the action head) with the clone's weights. Nothing else is copied."""
    for target, source in zip(actor_modules(model.policy), actor_modules(clone.policy)): target.load_state_dict(source.state_dict())


def build_model(arm, seed, mix, clone=None):
    """PPO built exactly as `autobot.sim.cli train` builds it; fine-tuning arms then take the clone's actor."""
    import torch
    from stable_baselines3 import PPO
    from autobot.sim import cli
    torch.set_num_threads(1)
    model = PPO("MlpPolicy", training_env(mix, seed), device="cpu", seed=seed, verbose=0, n_steps=ROLLOUT, batch_size=32, gamma=GAMMA)
    if arm != "scratch": import_actor(model, clone)
    setattr(model, cli.OBSERVATION_METADATA_ATTRIBUTE, cli.observation_metadata(PROFILE, ACTION_PROFILE, ENCODING))
    return model


def set_actor_trainable(model, trainable):
    for parameter in parameters_of(actor_modules(model.policy)): parameter.requires_grad_(bool(trainable))


def fresh_optimizer(model):
    """Re-create the policy optimiser as PPO first built it (Adam, same learning rate and epsilon, no moments)."""
    policy = model.policy
    policy.optimizer = policy.optimizer_class(policy.parameters(), lr=model.lr_schedule(1.0), **policy.optimizer_kwargs)


def action_probabilities(model, observations):
    import torch
    with torch.no_grad():
        return model.policy.get_distribution(torch.as_tensor(observations, dtype=torch.float32)).distribution.probs.numpy()


def state_values(model, observations):
    import torch
    with torch.no_grad(): return model.policy.predict_values(torch.as_tensor(observations, dtype=torch.float32)).numpy().reshape(-1)


def probability_check(model, reference, observations):
    probabilities = action_probabilities(model, observations)
    return {"observations": int(len(observations)), "max_abs_probability_difference": float(np.abs(probabilities - reference).max()),
            "identical": bool(np.array_equal(probabilities, reference)),
            "same_greedy_tactic": float((probabilities.argmax(1) == reference.argmax(1)).mean())}


# ---------------------------------------------------------------- evaluation

def end_pit(env, outcome):
    """Which pit the episode ended in: red's for a win, blue's for an own fall."""
    if outcome not in ("wins", "own_falls"): return None
    position = env.sim.truth(1 if outcome == "wins" else 0)["position_m"][:2]
    pits = np.asarray(env.sim.cfg["arena"]["pit_centres_m"], dtype=float)
    return PIT_NAMES[int(np.argmin(np.linalg.norm(pits - position, axis=1)))]


def run_sets(actor, env, sets):
    """{name: (starts, seeds)} -> {name: [episodes]}, with the episode seeds of `ladder.run_starts` (seed + start index)."""
    from autobot.sim.ladder import ladder_rollout
    results = {}
    for name, (starts, seeds) in sets.items():
        rows = []
        for seed in seeds:
            for index, start in enumerate(starts):
                episode = ladder_rollout(actor, env, start, int(seed) + index)
                rows.append(dict(episode, end_pit=end_pit(env, episode["outcome"])))
        results[name] = rows
    return results


def evaluation_sets(ladder, confirmation, full):
    """Monitor: hold-out starts once, ordinary transfer starts on three seeds. Full: every seed, plus the confirmation starts.

    The monitor episodes are a subset of the full ones (same starts, same seeds), so curves and final tables match.
    """
    sets = {"holdout": (ladder["holdout"], FINAL_SEEDS if full else FINAL_SEEDS[:1]), "transfer": (experiment.transfer_scenarios(), FINAL_SEEDS)}
    if full: sets["confirmation"] = (confirmation, FINAL_SEEDS)
    return sets


def load_actor(spec):
    """(policy with .predict, checkpoint metadata or None)."""
    import torch
    torch.set_num_threads(1)
    from autobot.sim.cli import checkpoint_observation_metadata, checkpoint_ppo_settings, load_checkpoint
    from autobot.sim.ladder import GreedyActor
    if spec["kind"] != "checkpoint": return experiment.reference_policy(spec["kind"]), None
    model, encoding = load_checkpoint(spec["path"], device="cpu"); metadata = checkpoint_observation_metadata(spec["path"])
    assert encoding == ENCODING and metadata["action_profile"] == ACTION_PROFILE and metadata["observation_profile"] == PROFILE, spec["path"]
    settings = checkpoint_ppo_settings(spec["path"])
    return GreedyActor(model), {"path": experiment.relative(spec["path"]), "gamma": settings["gamma"], "num_timesteps": settings["num_timesteps"]}


def _evaluate(task):
    name, spec, full, cache = task
    if cache and Path(cache).is_file(): return name, json.loads(Path(cache).read_text(encoding="utf-8"))
    ladder = experiment.load_ladder(); _, confirmation = load_mix()
    actor, metadata = load_actor(spec); started = time.perf_counter()
    result = {"kind": spec["kind"], **(metadata or {}), **{key: spec[key] for key in ("arm", "seed", "step", "actor_updates") if key in spec},
              "episodes": run_sets(actor, experiment.make_env(), evaluation_sets(ladder, confirmation, full))}
    result["wall_seconds"] = time.perf_counter() - started
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True); Path(cache).write_text(json.dumps(result), encoding="utf-8")
    return name, result


def cell(episodes):
    counts = Counter(episode["outcome"] for episode in episodes)
    wins = [episode for episode in episodes if episode["outcome"] == "wins"]
    return {"episodes": len(episodes), "wins": counts["wins"], "own_falls": counts["own_falls"], "timeouts": counts["timeouts"],
            "draws": counts["draws"], "simulation_bugs": counts["simulation_bugs"],
            "wins_with_a_turn_before_contact": sum(episode["turn_decisions_before_contact"] > 0 for episode in wins)}


def needs_turn(episode, first_gated):
    return abs(episode["heading_offset_deg"]) >= first_gated


def tables(episodes, first_gated):
    """Per-pit turn-needed table and ordinary-start tables for one policy."""
    turn = [episode for episode in episodes["holdout"] if needs_turn(episode, first_gated)]
    table = {"turn_needed": {"all": cell(turn), **{pit: cell([e for e in turn if e["pit"] == pit]) for pit in PIT_NAMES}},
             "ordinary": {"all": cell(episodes["transfer"])}}
    for group in sorted({episode["group"] for episode in episodes["transfer"]}):
        table["ordinary"][group] = cell([e for e in episodes["transfer"] if e["group"] == group])
    ordinary = list(episodes["transfer"])
    if "confirmation" in episodes:
        table["ordinary"]["new_confirmation"] = cell(episodes["confirmation"]); ordinary += episodes["confirmation"]
    table["ordinary_end_pit"] = {pit: {"wins": sum(e["outcome"] == "wins" and e["end_pit"] == pit for e in ordinary),
                                       "own_falls": sum(e["outcome"] == "own_falls" and e["end_pit"] == pit for e in ordinary)} for pit in PIT_NAMES}
    return table


def recommendation(candidate, clone, advance):
    """Apply RULE to one candidate's tables. Returns the checks and whether the checkpoint may be recommended."""
    turn = {scope: candidate["turn_needed"][scope]["wins"] >= RULE["turn_share"] * clone["turn_needed"][scope]["wins"]
            and candidate["turn_needed"][scope]["wins"] > 0 for scope in ("all", *PIT_NAMES)}
    new = candidate["ordinary"]["new_confirmation"]; margin = RULE["ordinary_margin"] * new["episodes"]
    checks = {"keeps_turning_overall": turn["all"], **{f"keeps_turning_{pit}": turn[pit] for pit in PIT_NAMES},
              "beats_clone_on_new_confirmation": new["wins"] - clone["ordinary"]["new_confirmation"]["wins"] >= margin,
              "beats_advance_on_new_confirmation": new["wins"] - advance["ordinary"]["new_confirmation"]["wins"] >= margin,
              "no_more_own_falls_than_clone_on_new_confirmation": new["own_falls"] <= clone["ordinary"]["new_confirmation"]["own_falls"],
              "beats_clone_on_fresh_normal": candidate["ordinary"]["fresh_normal"]["wins"] > clone["ordinary"]["fresh_normal"]["wins"],
              "beats_advance_on_fresh_normal": candidate["ordinary"]["fresh_normal"]["wins"] > advance["ordinary"]["fresh_normal"]["wins"]}
    return {"checks": {key: bool(value) for key, value in checks.items()}, "recommend": bool(all(checks.values()))}


# ---------------------------------------------------------------- import verification

def saved_observations():
    """Fixed observations for the probability check: controller-driven hold-out episodes and clone-driven ordinary starts."""
    if OBSERVATIONS_PATH.is_file():
        data = np.load(OBSERVATIONS_PATH); return data["observations"], {int(seed): data[f"clone{seed}"] for seed in PAIRS.values()}
    import torch
    torch.set_num_threads(1)
    import controller_imitation as imitation
    from autobot.sim.cli import load_checkpoint
    from autobot.sim.ladder import GreedyActor
    ladder = experiment.load_ladder(); recorder = imitation.Recorder(experiment.make_env())
    clones = {seed: load_checkpoint(clone_path(seed), device="cpu")[0] for seed in PAIRS.values()}
    rows = [recorder.episode(None, start, imitation.DATA_SEED + 50_000 + index)[0] for index, start in enumerate(ladder["holdout"])]
    driver = GreedyActor(clones[0])
    rows += [recorder.episode(driver, start, imitation.DATA_SEED + 70_000 + index, 300)[0] for index, start in enumerate(experiment.transfer_scenarios())]
    observations = np.concatenate(rows).astype(np.float32)
    reference = {seed: action_probabilities(model, observations) for seed, model in clones.items()}
    np.savez_compressed(OBSERVATIONS_PATH, observations=observations, **{f"clone{seed}": value for seed, value in reference.items()})
    return observations, reference


def _verify_one(task):
    seed, clone_seed = task
    import tempfile
    import torch
    torch.set_num_threads(1)
    from autobot.sim.cli import checkpoint_observation_metadata, load_checkpoint
    from autobot.sim.ladder import GreedyActor
    observations, reference = saved_observations(); mix, _ = load_mix(); ladder = experiment.load_ladder()
    clone, _ = load_checkpoint(clone_path(clone_seed), device="cpu")
    scratch = build_model("scratch", seed, mix); model = build_model("finetune", seed, mix, clone)
    row = {"ppo_seed": seed, "clone": experiment.relative(clone_path(clone_seed)),
           "clone_file_reproduces_saved_probabilities": probability_check(clone, reference[clone_seed], observations),
           "imported_into_fresh_ppo": probability_check(model, reference[clone_seed], observations),
           "scratch_arm_same_seed_differs": not np.array_equal(action_probabilities(scratch, observations), reference[clone_seed]),
           "actor_weights_identical_to_clone": bool(np.array_equal(flat(actor_modules(model.policy)), flat(actor_modules(clone.policy)))),
           "critic": {"identical_to_scratch_arm_of_same_seed": bool(np.array_equal(flat(critic_modules(model.policy)), flat(critic_modules(scratch.policy)))),
                      "differs_from_the_clone_files_unused_critic": not np.array_equal(flat(critic_modules(model.policy)), flat(critic_modules(clone.policy))),
                      "value_on_saved_observations": {"mean": float(state_values(model, observations).mean()), "std": float(state_values(model, observations).std())}},
           "optimizer": {"class": type(model.policy.optimizer).__name__, "learning_rate": float(model.policy.optimizer.param_groups[0]["lr"]),
                         "eps": float(model.policy.optimizer.param_groups[0]["eps"]),
                         "has_no_state_before_training": len(model.policy.optimizer.state_dict()["state"]) == 0}}
    with tempfile.TemporaryDirectory() as directory:
        model.save(Path(directory) / "imported"); reloaded, encoding = load_checkpoint(Path(directory) / "imported.zip", device="cpu")
        row["after_save_and_reload"] = dict(probability_check(reloaded, reference[clone_seed], observations), previous_action_encoding=encoding,
                                            metadata=checkpoint_observation_metadata(Path(directory) / "imported.zip"))
    # Closed loop: the imported actor, greedy, on the starts and seeds of the saved clone baseline.
    episodes = run_sets(GreedyActor(model), experiment.make_env(), {"holdout": (ladder["holdout"], FINAL_SEEDS),
                                                                   "transfer": (experiment.transfer_scenarios(), FINAL_SEEDS)})
    saved = json.loads(LADDER_RESULTS_PATH.read_text(encoding="utf-8"))["results"][f"supervised_mlp_64_64_seed{clone_seed}_cloning_only"]
    closed = {}
    for name in ("holdout", "transfer"):
        key = lambda episode: (episode["scenario"], episode["seed"])
        before = {key(episode): (episode["outcome"], episode["steps"]) for episode in saved[name]}
        now = {key(episode): (episode["outcome"], episode["steps"]) for episode in episodes[name]}
        closed[name] = {"episodes": len(now), "saved_baseline_episodes": len(before), "same_episode_keys": set(now) == set(before),
                        "same_outcome_and_length": sum(now[k] == before.get(k) for k in now),
                        "imported": cell(episodes[name]), "saved_baseline": dict(Counter(outcome for outcome, _ in before.values()))}
    row["closed_loop_against_saved_baseline"] = closed
    row["passed"] = bool(row["imported_into_fresh_ppo"]["identical"] and row["after_save_and_reload"]["identical"]
                         and row["actor_weights_identical_to_clone"] and row["critic"]["identical_to_scratch_arm_of_same_seed"]
                         and all(part["same_episode_keys"] and part["same_outcome_and_length"] == part["episodes"] for part in closed.values()))
    return seed, row


def verify(args):
    observations, _ = saved_observations()
    with ProcessPoolExecutor(max_workers=len(PAIRS)) as pool: rows = dict(pool.map(_verify_one, list(PAIRS.items())))
    output = {"purpose": "Before any update: does importing the behaviour-cloned actor into a fresh PPO preserve its action "
                         "probabilities on saved observations and reproduce its saved closed-loop baseline?",
              "saved_observations": {"file": experiment.relative(OBSERVATIONS_PATH), "count": int(len(observations)),
                                     "source": "controller-driven episodes from the ladder hold-out starts and clone-driven "
                                               "episodes from the ordinary transfer starts"},
              "saved_closed_loop_baseline": experiment.relative(LADDER_RESULTS_PATH) + " (supervised_mlp_64_64_seed*_cloning_only)",
              "initialisation": __doc__.split("Critic and optimiser.")[1].strip(), "runs": rows, "passed": all(row["passed"] for row in rows.values())}
    IMPORT_CHECK_PATH.write_text(json.dumps(output, indent=1), encoding="utf-8"); print(IMPORT_CHECK_PATH)
    for seed, row in rows.items():
        closed = row["closed_loop_against_saved_baseline"]
        print(f"seed {seed}: probabilities identical {row['imported_into_fresh_ppo']['identical']} (max diff "
              f"{row['imported_into_fresh_ppo']['max_abs_probability_difference']:.1e}), reload identical {row['after_save_and_reload']['identical']}, "
              + ", ".join(f"{name} {part['same_outcome_and_length']}/{part['episodes']} episodes reproduced" for name, part in closed.items())
              + f" -> {'PASS' if row['passed'] else 'FAIL'}")
    if not output["passed"]: raise SystemExit("Import verification failed; do not train.")


# ---------------------------------------------------------------- training

def train(args):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from autobot.sim import cli
    from autobot.sim.core import load_config
    from autobot.sim.env import ACTION_PROFILES
    from autobot.sim.ladder import TURN_TACTICS
    arm, seed = args.arm, args.seed
    out = Path(args.output) if args.output else run_directory(arm, seed)
    if out.exists() and any(out.iterdir()): raise SystemExit(f"Refusing to overwrite non-empty run directory: {out}")
    total = args.total_steps; warmup = (args.warmup_steps if arm == "finetune" else 0)
    if total % ROLLOUT or warmup % ROLLOUT or warmup >= total: raise SystemExit("Step counts must be multiples of 256 with warm-up below the total")
    schedule = checkpoint_steps(arm) if (total, args.warmup_steps) == (TOTAL_STEPS, WARMUP_STEPS) else {0: 0, warmup: 0, total: (total - warmup) // ROLLOUT}
    mix, _ = load_mix(); tactics = ACTION_PROFILES[ACTION_PROFILE]["tactics"]
    clone = None; clone_file = None; import_check = None
    if arm != "scratch":
        clone_file = Path(args.clone) if args.clone else clone_path(PAIRS[seed])
        clone, _ = cli.load_checkpoint(clone_file, device="cpu")
    model = build_model(arm, seed, mix, clone)
    if clone is not None:
        observations, reference = saved_observations()
        own = action_probabilities(clone, observations)
        import_check = probability_check(model, own, observations)
        if not import_check["identical"]: raise SystemExit(f"Imported actor does not reproduce the clone's probabilities: {import_check}")
    clone_actor = None if clone is None else flat(actor_modules(clone.policy))
    actual = cli.ppo_hyperparameters(model)
    if actual["gamma"] != GAMMA or actual["rollout_buffer_gamma"] != GAMMA: raise SystemExit(f"PPO is not using gamma={GAMMA}: {actual}")
    out.mkdir(parents=True, exist_ok=True); (out / "checkpoints").mkdir()
    initialisation = {"actor": "random (PPO default)" if clone is None else f"copied from {path_label(clone_file)}",
                      "critic": "random (PPO default for this seed)" + (f", then {warmup} steps of critic-only updates with the actor frozen" if warmup else ""),
                      "optimizer": "Adam as PPO builds it (lr 3e-4, eps 1e-5), no state" + ("; re-created with no state when actor updates begin" if warmup else "")}
    metadata = {"trainer": "scripts/clone_finetune.py", "arm": arm, "config": load_config(str(CONFIG)), "config_path": str(CONFIG.resolve()),
                "seed": seed, "profile": PROFILE, "n_envs": 1, "opponent_mode": "scripted", "opponent_chassis": "last_year",
                "reward_profile": "outcome_only", "action_profile": ACTION_PROFILE, "previous_action_encoding": ENCODING, "gamma": GAMMA,
                "action_names": list(tactics), "observation_normalization": False,
                "observation_fields": cli.observation_fields(PROFILE, ACTION_PROFILE, ENCODING),
                "training_curriculum": dict(mix, seed=int(mix["seed"]) + 1000 * seed), "mix_path": experiment.relative(MIX_PATH),
                "total_timesteps": total, "critic_warmup_timesteps": warmup, "checkpoint_steps": {str(k): v for k, v in schedule.items()},
                "initialisation": initialisation, "import_check": import_check, "ppo_hyperparameters": actual}
    (out / "effective_config.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    metrics_path = out / "metrics.jsonl"

    def log(record):
        with metrics_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
    log({"type": "run_start", "arm": arm, "seed": seed, "previous_action_encoding": ENCODING, "action_profile": ACTION_PROFILE,
         "profile": PROFILE, "ppo_hyperparameters": actual, "initialisation": initialisation, "import_check": import_check})

    class PilotCallback(BaseCallback):
        def __init__(self):
            super().__init__(); self.counts = Counter(); self.first_contact = None; self.pre_turns = 0; self.episode_index = 0
            self.warming = warmup > 0; self.started = time.perf_counter()

        def _on_training_start(self):
            if self.warming: set_actor_trainable(self.model, False)

        def _on_rollout_start(self):
            steps = self.num_timesteps; values = self.model.logger.name_to_value
            if steps:
                # Stable-Baselines3's own numbers for the update that just finished.
                log({"type": "update", "timesteps": steps, "actor_frozen": self.warming,
                     **{key.split("/")[1]: float(values[key]) for key in ("train/approx_kl", "train/clip_fraction", "train/entropy_loss",
                                                                          "train/value_loss", "train/explained_variance",
                                                                          "train/policy_gradient_loss") if key in values}})
            if self.warming and steps >= warmup:
                unchanged = bool(np.array_equal(flat(actor_modules(self.model.policy)), clone_actor))
                set_actor_trainable(self.model, True); fresh_optimizer(self.model); self.warming = False
                log({"type": "critic_warmup_complete", "timesteps": steps, "actor_unchanged_by_warmup": unchanged,
                     "optimizer": "re-created, no state", "elapsed_seconds": time.perf_counter() - self.started})
                if not unchanged: raise RuntimeError("Critic warm-up changed the actor")
            if steps in schedule:
                path = out / "checkpoints" / f"step_{steps:07d}"; self.model.save(path)
                log({"type": "checkpoint", "timesteps": steps, "actor_updates": schedule[steps], "path": f"checkpoints/{path.name}.zip"})

        def _on_step(self):
            action = int(np.asarray(self.locals["actions"]).reshape(-1)[0]); info = self.locals["infos"][0]
            tactic = tactics[action]; self.counts[tactic] += 1
            if self.first_contact is None:
                self.pre_turns += tactic in TURN_TACTICS
                if info.get("opponent_contact"): self.first_contact = sum(self.counts.values())
            if bool(np.asarray(self.locals["dones"]).reshape(-1)[0]):
                reason = info.get("round_end_reason"); bot = info.get("round_end_bot")
                outcome = ("own_fall" if reason == "Pit fall" and bot == 0 else "win" if reason == "Pit fall" and bot == 1
                           else "draw" if reason == "Simultaneous pit fall" else "timeout" if reason == "Timeout" else "simulation_bug")
                log({"type": "train_episode", "timesteps": self.num_timesteps, "episode_index": self.episode_index,
                     "group": info.get("scenario_group"), "scenario": info.get("scenario"), "outcome": outcome, "actor_frozen": self.warming,
                     "steps": sum(self.counts.values()), "first_contact_step": self.first_contact,
                     "turn_decisions_before_contact": self.pre_turns, "actions": dict(self.counts)})
                self.counts = Counter(); self.first_contact = None; self.pre_turns = 0; self.episode_index += 1
            return True

    callback = PilotCallback(); started = time.perf_counter()
    model.learn(total, callback=callback)
    elapsed = time.perf_counter() - started
    model.save(out / "checkpoints" / f"step_{total:07d}"); model.save(out / "final_model")
    log({"type": "checkpoint", "timesteps": total, "actor_updates": schedule[total], "path": f"checkpoints/step_{total:07d}.zip"})
    summary = {"type": "run_complete", "arm": arm, "ending_timesteps": int(model.num_timesteps), "episodes": callback.episode_index,
               "elapsed_seconds": elapsed, "environment_steps_per_second": model.num_timesteps / elapsed}
    log(summary); print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- monitoring and the final comparison

def run_checkpoints(arm, seed):
    """[(step, actor updates, path)] of the checkpoints a run has saved so far."""
    run = run_directory(arm, seed); schedule = checkpoint_steps(arm)
    return [(step, updates, run / "checkpoints" / f"step_{step:07d}.zip") for step, updates in schedule.items()
            if (run / "checkpoints" / f"step_{step:07d}.zip").is_file()]


def monitored(arm, updates, step):
    return arm != "scratch" or updates in SCRATCH_MONITOR_UPDATES or step == TOTAL_STEPS


def monitor(args):
    tasks = []
    for arm in ARMS:
        for seed in PAIRS:
            for step, updates, path in run_checkpoints(arm, seed):
                if not monitored(arm, updates, step): continue
                cache = run_directory(arm, seed) / "monitor" / f"step_{step:07d}.json"
                if not cache.is_file():
                    tasks.append((f"{arm}/seed{seed}/step{step}", {"kind": "checkpoint", "path": str(path), "arm": arm, "seed": seed,
                                                                   "step": step, "actor_updates": updates}, False, str(cache)))
    print(f"{len(tasks)} checkpoints to evaluate", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, result in pool.map(_evaluate, tasks): print(name, f"{result['wall_seconds']:.0f}s", flush=True)


def monitor_curves(first_gated):
    curves = {}
    for arm in ARMS:
        for seed in PAIRS:
            rows = []
            for step, updates, _ in run_checkpoints(arm, seed):
                cache = run_directory(arm, seed) / "monitor" / f"step_{step:07d}.json"
                if not cache.is_file(): continue
                table = tables(json.loads(cache.read_text(encoding="utf-8"))["episodes"], first_gated)
                rows.append({"timesteps": step, "actor_updates": updates, **table})
            curves[f"{arm}/seed{seed}"] = rows
    return curves


def training_record(arm, seed):
    """What the run logged: sampled (exploring) training episodes by group and phase, and PPO's own update statistics."""
    path = run_directory(arm, seed) / "metrics.jsonl"
    if not path.is_file(): return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    config = json.loads((run_directory(arm, seed) / "effective_config.json").read_text(encoding="utf-8"))
    warmup = config["critic_warmup_timesteps"]; total = config["total_timesteps"]
    edges = [0] + ([warmup] if warmup else []) + [warmup + (total - warmup) * quarter // 4 for quarter in (1, 2, 3, 4)]
    episodes = [row for row in rows if row["type"] == "train_episode"]; updates = [row for row in rows if row["type"] == "update"]
    phases = []
    for low, high in zip(edges, edges[1:]):
        part = [row for row in episodes if low < row["timesteps"] <= high]; groups = {}
        for row in part:
            entry = groups.setdefault(row["group"], Counter()); entry[row["outcome"]] += 1; entry["episodes"] += 1; entry["steps"] += row["steps"]
            entry["episodes_with_a_turn_before_contact"] += row["turn_decisions_before_contact"] > 0
        tactic = Counter()
        for row in part: tactic.update(row["actions"])
        window = [row for row in updates if low < row["timesteps"] <= high]
        mean = lambda key: float(np.mean([row[key] for row in window if key in row])) if any(key in row for row in window) else None
        median = lambda key: float(np.median([row[key] for row in window if key in row])) if any(key in row for row in window) else None
        phases.append({"timesteps": [low, high], "actor_frozen": bool(warmup and high <= warmup),
                       "sampled_training_episodes": {group: dict(entry) for group, entry in sorted(groups.items())},
                       "sampled_tactic_share": {name: round(count / max(sum(tactic.values()), 1), 4) for name, count in sorted(tactic.items())},
                       "mean_policy_entropy": None if mean("entropy_loss") is None else -mean("entropy_loss"),
                       "mean_approx_kl": mean("approx_kl"), "mean_clip_fraction": mean("clip_fraction"),
                       # Per-rollout explained variance has huge negative outliers when a rollout's returns barely vary: use the median.
                       "median_explained_variance": median("explained_variance"), "mean_value_loss": mean("value_loss")})
    warm = next((row for row in rows if row["type"] == "critic_warmup_complete"), None)
    last_warm = [row for row in updates if warmup and warmup - 40 * ROLLOUT < row["timesteps"] <= warmup and "explained_variance" in row]
    first_warm = [row for row in updates if warmup and row["timesteps"] <= 10 * ROLLOUT and "value_loss" in row]
    return {"complete": next((row for row in rows if row["type"] == "run_complete"), None), "initialisation": config["initialisation"],
            "import_check": config["import_check"], "critic_warmup": warm and dict(warm, median_explained_variance_last_40_updates=float(np.median([row["explained_variance"] for row in last_warm])),
                                                           mean_value_loss_first_10_updates=float(np.mean([row["value_loss"] for row in first_warm])),
                                                           mean_value_loss_last_40_updates=float(np.mean([row["value_loss"] for row in last_warm]))),
            "ppo_hyperparameters": config["ppo_hyperparameters"], "training_episodes": len(episodes), "phases": phases}


def evaluate(args):
    ladder = experiment.load_ladder(); mix, confirmation = load_mix()
    first_gated = min(rung["abs_heading_offset_deg"] for rung in ladder["rungs"] if not rung["warmup"])
    cache = lambda name: str(ROOT / "runs/clone-pilot-final-evaluation" / f"{name.replace('/', '_')}.json")
    tasks = [(name, {"kind": name}, True, cache(name)) for name in ("always_advance", "turn_then_advance_controller")]
    for seed, clone_seed in PAIRS.items():
        tasks.append((f"frozen_clone/seed{seed}", {"kind": "checkpoint", "path": str(clone_path(clone_seed)), "arm": "frozen_clone", "seed": seed}, True,
                      cache(f"frozen_clone/seed{seed}")))
        for arm in ARMS:
            path = run_directory(arm, seed) / "final_model.zip"
            if path.is_file():
                tasks.append((f"{arm}/seed{seed}", {"kind": "checkpoint", "path": str(path), "arm": arm, "seed": seed, "step": TOTAL_STEPS}, True,
                              cache(f"{arm}/seed{seed}")))
    with ProcessPoolExecutor(max_workers=args.workers) as pool: results = dict(pool.map(_evaluate, tasks))
    final = {name: tables(result["episodes"], first_gated) for name, result in results.items()}
    verdicts = {}
    for seed in PAIRS:
        for arm in ("finetune", "finetune_cold_critic"):
            if f"{arm}/seed{seed}" in final:
                verdicts[f"{arm}/seed{seed}"] = recommendation(final[f"{arm}/seed{seed}"], final[f"frozen_clone/seed{seed}"], final["always_advance"])
    output = {
        "purpose": "Bounded pilot: does PPO keep, lose or extend turn-then-push when its actor starts as the behaviour-cloned controller copy?",
        "fixed": {"config": experiment.relative(CONFIG), "profile": PROFILE, "action_profile": ACTION_PROFILE, "previous_action_encoding": ENCODING,
                  "gamma": GAMMA, "reward_profile": "outcome_only", "opponent": "scripted last_year", "environment": "AutoBotEnv, unchanged"},
        "design": {"arms": {"frozen_clone": "plain behaviour-cloned 64-64 actor, never updated", "finetune": "clone actor, critic warm-up, then PPO",
                            "scratch": "PPO from random initialisation", "finetune_cold_critic": "extra control: clone actor, random critic, PPO from step 0"},
                   "seed_pairs_ppo_to_clone": {str(seed): experiment.relative(clone_path(clone)) for seed, clone in PAIRS.items()},
                   "environment_steps_per_training_arm": TOTAL_STEPS, "critic_warmup_steps_finetune": WARMUP_STEPS,
                   "actor_updates": {"finetune": PPO_STEPS // ROLLOUT, "finetune_cold_critic": TOTAL_STEPS // ROLLOUT, "scratch": TOTAL_STEPS // ROLLOUT},
                   "initialisation": __doc__.split("Critic and optimiser.")[1].strip(),
                   "training_mix": {"file": experiment.relative(MIX_PATH), "group_weights": mix["stages"][0]["group_weights"],
                                    "starts": {name: len(group) for name, group in mix["groups"].items()}, "provenance": mix["provenance"],
                                    "start_order": "episode-indexed, depends on the mix and the seed only, so it is the same for every arm of a seed"},
                   "never_trained_on": {"turn_needed": f"ladder hold-out starts at {first_gated:g} degrees or more (108 starts)",
                                        "ordinary": "ordinary_training (4), frozen_normal (4) and fresh_normal (12) starts of the earlier transfer test",
                                        "new_confirmation": f"{len(confirmation)} new ordinary starts ({experiment.relative(CONFIRMATION_PATH)}), "
                                                            "run only in this final evaluation"},
                   "seeds": {"final": list(FINAL_SEEDS), "monitor": "hold-out starts on the first final seed, ordinary transfer starts on all three; "
                                                                    "a subset of the final episodes"},
                   "greedy": "all evaluation is greedy (argmax); training samples from the policy"},
        "rule": RULE, "import_check": json.loads(IMPORT_CHECK_PATH.read_text(encoding="utf-8")) if IMPORT_CHECK_PATH.is_file() else None,
        "final": final, "recommendation": verdicts,
        "any_checkpoint_recommended": any(v["recommend"] for name, v in verdicts.items() if name.startswith("finetune/")),
        "monitor": monitor_curves(first_gated),
        "training": {f"{arm}/seed{seed}": training_record(arm, seed) for arm in ARMS for seed in PAIRS},
        "results": results}
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8"); print(args.output)
    for name, table in final.items():
        turn = table["turn_needed"]; ordinary = table["ordinary"]
        show = lambda c: f"{c['wins']}/{c['episodes']} (falls {c['own_falls']}, timeouts {c['timeouts']})"
        print(f"{name:32s} turn TL {show(turn['top_left'])} BR {show(turn['bottom_right'])} | ordinary {show(ordinary['all'])} "
              f"| new confirmation {show(ordinary['new_confirmation'])}" + (f" | recommend {verdicts[name]['recommend']}" if name in verdicts else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make"); sub.add_parser("verify")
    t = sub.add_parser("train"); t.add_argument("--arm", choices=ARMS, required=True); t.add_argument("--seed", type=int, required=True)
    t.add_argument("--clone", help="behaviour-cloned checkpoint (default: the clone paired with the seed)"); t.add_argument("--output")
    t.add_argument("--total-steps", type=int, default=TOTAL_STEPS); t.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    m = sub.add_parser("monitor"); m.add_argument("--workers", type=int, default=14)
    e = sub.add_parser("evaluate"); e.add_argument("--workers", type=int, default=14)
    e.add_argument("--output", default=str(ROOT / "artifacts/clone-finetune-comparison.json"))
    args = parser.parse_args()
    {"make": make, "verify": verify, "train": train, "monitor": monitor, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
