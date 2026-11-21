"""Can one supervised 64-64 clone hold both skills: the turning drill and ordinary starts? Simulation-only; no PPO.

    python scripts/combined_clone.py make       # new ordinary demonstration and validation starts (written once)
    python scripts/combined_clone.py collect    # controller demonstrations: the original drill set plus the new ordinary starts
    python scripts/combined_clone.py train      # fresh 64-64 clones: three recipes x three seeds
    python scripts/combined_clone.py validate   # closed loop on validation starts; picks the recipe
    python scripts/combined_clone.py evaluate   # held-out comparison and the keep-as-demo rule

The teacher is the existing scripted turn-then-advance controller (FlankController). It reads simulator truth, but
everything it uses (both positions, own heading, the fixed pits) is in the sim_debug observation, so the student
sees enough to copy it. Students are the network PPO trains (SB3 MlpPolicy actor, two tanh layers of 64) fitted by
cross-entropy; the value head is never touched and nothing here runs PPO.

Recipes (all start from the same fresh network for a seed):
    concat           every demonstration decision in one pool, sampled uniformly (the literal "combine")
    balanced         each minibatch is half drill decisions, half ordinary decisions
    balanced_dagger  `balanced`, then DAgger: the student drives the training starts, the controller labels the
                     states it reaches, and the student is refitted on everything

The recipe is chosen on validation starts only. Held-out starts (ladder hold-out, the earlier ordinary starts and
the ordinary confirmation starts) are never used for fitting or for that choice.
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
import clone_finetune as pilot  # noqa: E402
import controller_imitation as imitation  # noqa: E402
import heading_ladder as experiment  # noqa: E402
import ordinary_start_diagnostic as diagnostic  # noqa: E402

ROOT = experiment.ROOT
STARTS_PATH = ROOT / "config/combined_clone_starts_v1.json"
DATA_DIR = ROOT / "runs/combined-clone-data"
EVALUATION_DIR = ROOT / "runs/combined-clone-evaluation"
REFERENCE_DIR = ROOT / "runs/clone-pilot-final-evaluation"      # the same held-out starts and seeds, evaluated in the pilot
VALIDATION_PATH = ROOT / "artifacts/combined-clone-validation.json"
STARTS_SEED = 97017
TRAIN_PER_ARCHETYPE = 30; VALIDATION_PER_ARCHETYPE = 10
SEEDS = (0, 1, 2)                               # the seeds of the drill-only clones
RECIPES = ("concat", "balanced", "balanced_dagger")
BATCH = 256; LEARNING_RATE = 1e-3; BC_STEPS = 20_000
DAGGER_ITERATIONS = 4; DAGGER_STEPS = 8_000
DRILL_STEP_CAP = imitation.STUDENT_STEP_CAP; ORDINARY_STEP_CAP = 600      # decisions per student-driven DAgger episode
ORDINARY_DEMO_SEED = imitation.DATA_SEED + 200_000
VALIDATION_DEMO_SEED = imitation.DATA_SEED + 300_000
DAGGER_SEED = imitation.DATA_SEED + 400_000
VALIDATION_SEED = experiment.VALIDATION_SEEDS[0]
SOURCES = ("drill", "ordinary", "ordinary_validation")
# Both rules were fixed before any clone was trained.
SELECTION = {"data": "validation starts only: ladder validation starts that need a turn, and new ordinary validation starts",
             "eligible": "wins at least 90% of the turn-needed validation episodes at each pit, pooled over the three seeds",
             "score": "ordinary validation wins minus own falls, pooled over the three seeds; ties go to the simpler recipe",
             "turn_share": 0.9}
KEEP = {"applies_to": "each seed of the selected recipe, on held-out starts",
        "keeps_drill": "at least 90% of the drill-only clone's turn-needed wins at each pit",
        "wins_ordinary": "on both ordinary sets: more wins than Always-Advance and at least half the controller's wins",
        "does_not_fall": "own falls on at most 5% of episodes of each ordinary set",
        "verdict": "worth keeping as a simulator demo if at least two of the three seeds pass",
        "turn_share": 0.9, "controller_share": 0.5, "max_fall_share": 0.05}
ORDINARY_SETS = ("earlier_ordinary", "new_confirmation")


def run_directory(recipe, seed):
    return ROOT / f"runs/combined-clone-{recipe.replace('_', '-')}-seed{seed}"


# ---------------------------------------------------------------- starts

def build_starts(ladder, arena):
    """New ordinary starts for demonstrations and for validation; none repeats a held-out case."""
    from autobot.sim.scenarios import load_scenario_file
    archetypes = load_scenario_file(diagnostic.FROZEN / "training_curriculum.json")["groups"]["ordinary"]
    _, confirmation = pilot.load_mix(); held = pilot.heldout_starts(ladder) + confirmation
    rng = np.random.default_rng(STARTS_SEED)
    train = pilot.jittered(archetypes, TRAIN_PER_ARCHETYPE, rng, "demo", "ordinary_demonstration", held, arena)
    validation = pilot.jittered(archetypes, VALIDATION_PER_ARCHETYPE, rng, "demo_validation", "ordinary_validation", held + train, arena)
    starts = {"schema": "combined_clone_starts_v1", "seed": STARTS_SEED,
              "provenance": f"seeded offsets (+-{pilot.JITTER_M} m, +-{pilot.JITTER_RAD} rad) of the four ordinary starts of the frozen "
                            "training curriculum; none is within the same-case tolerance of a held-out start",
              "same_case_tolerance": pilot.SAME_CASE, "train": train, "validation": validation}
    check_starts(starts, held)
    return starts


def check_starts(starts, held):
    """Demonstration starts repeat no held-out or validation case; validation starts repeat no held-out case."""
    clashes = [(a["name"], b["name"]) for a in starts["train"] for b in list(held) + starts["validation"] if pilot.same_case(a, b)]
    clashes += [(a["name"], b["name"]) for a in starts["validation"] for b in held if pilot.same_case(a, b)]
    if clashes: raise ValueError(f"Starts repeat held-out cases: {clashes[:3]}")


def load_starts():
    starts = json.loads(STARTS_PATH.read_text(encoding="utf-8"))
    _, confirmation = pilot.load_mix(); check_starts(starts, pilot.heldout_starts(experiment.load_ladder()) + confirmation)
    return starts


def drill_starts(ladder):
    """The starts of the original drill demonstrations: every ladder training start, warm-up rung included."""
    return [start for rung in ladder["rungs"] for start in rung["train"]]


def make(args):
    from autobot.sim.core import load_config
    if STARTS_PATH.exists(): raise SystemExit(f"{STARTS_PATH} already exists; not overwriting.")
    starts = build_starts(experiment.load_ladder(), load_config(str(experiment.CONFIG))["arena"])
    STARTS_PATH.write_text(json.dumps(starts, indent=1), encoding="utf-8")
    print(STARTS_PATH, {key: len(starts[key]) for key in ("train", "validation")})


# ---------------------------------------------------------------- demonstrations

def outcome_of(info):
    reason = info.get("round_end_reason"); bot = info.get("round_end_bot")
    if reason == "Pit fall": return "own_falls" if bot == 0 else "wins"
    return {"Simultaneous pit fall": "draws", "Timeout": "timeouts"}.get(reason, "simulation_bugs")


def demonstrate(env, expert, start, seed, driver=None, step_cap=None):
    """One episode driven by the controller (or by `driver`), every state labelled with the controller's choice.

    Controller-driven, this is exactly `controller_imitation.Recorder.episode`, plus how the episode ended.
    """
    observation, _ = env.reset(seed=int(seed), options={"scenario": start})
    observations, labels, taken = [], [], []; outcome = "cut"
    while True:
        label = int(expert.predict(observation)[0])
        action = label if driver is None else int(driver.predict(observation, deterministic=True)[0])
        observations.append(observation); labels.append(label); taken.append(action)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated: outcome = outcome_of(info); break
        if step_cap and len(labels) >= step_cap: break
    return np.asarray(observations, dtype=np.float32), np.asarray(labels), np.asarray(taken), outcome


def demonstration_plan(ladder, starts):
    """[(source, start, seed)]. The drill part uses the starts and seeds of the original behaviour-cloning data."""
    plan = [("drill", start, imitation.DATA_SEED + index) for index, start in enumerate(drill_starts(ladder))]
    plan += [("ordinary", start, ORDINARY_DEMO_SEED + index) for index, start in enumerate(starts["train"])]
    plan += [("ordinary_validation", start, VALIDATION_DEMO_SEED + index) for index, start in enumerate(starts["validation"])]
    return plan


def _collect_chunk(chunk):
    env = experiment.make_env(); expert = diagnostic.FlankController(); expert.bind(env); rows = []
    for position, source, start, seed in chunk:
        observations, labels, _, outcome = demonstrate(env, expert, start, seed)
        rows.append((position, source, start["name"], seed, observations, labels, outcome))
    return rows


def collect(args):
    if (DATA_DIR / "demonstrations.npz").exists(): raise SystemExit("Demonstrations already collected; not overwriting.")
    ladder = experiment.load_ladder(); starts = load_starts(); plan = demonstration_plan(ladder, starts)
    tasks = [(position, *item) for position, item in enumerate(plan)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = sorted((row for chunk in pool.map(_collect_chunk, [tasks[index::args.workers] for index in range(args.workers)]) for row in chunk),
                      key=lambda row: row[0])
    episodes = [{"episode": position, "source": source, "start": name, "seed": int(seed), "decisions": int(len(labels)), "outcome": outcome}
                for position, source, name, seed, _, labels, outcome in rows]
    tactics = experiment_tactics()
    summary = {}
    for source in SOURCES:
        part = [row for row in rows if row[1] == source]; labels = np.concatenate([row[5] for row in part])
        summary[source] = {"episodes": len(part), "decisions": int(len(labels)), "outcomes": dict(Counter(row[6] for row in part)),
                           "label_counts": {tactics[key]: int(value) for key, value in sorted(Counter(labels.tolist()).items())}}
    original = json.loads((ROOT / "artifacts/controller-imitation.json").read_text(encoding="utf-8"))["students"]["mlp_64_64_seed0"]["history"][0]
    same = (summary["drill"]["decisions"] == original["dataset_decisions"] and summary["drill"]["label_counts"] == original["label_counts"])
    summary["drill"]["reproduces_the_original_drill_demonstrations"] = bool(same)
    if not same: raise SystemExit(f"Drill demonstrations differ from the original data: {summary['drill']} vs {original['label_counts']}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / "demonstrations.npz", observations=np.concatenate([row[4] for row in rows]),
                        labels=np.concatenate([row[5] for row in rows]).astype(np.int8),
                        source=np.concatenate([np.full(len(row[5]), SOURCES.index(row[1]), dtype=np.int8) for row in rows]),
                        episode=np.concatenate([np.full(len(row[5]), row[0], dtype=np.int32) for row in rows]))
    (DATA_DIR / "demonstrations.json").write_text(json.dumps({"teacher": "ordinary_start_diagnostic.FlankController (scripted, reads simulator truth)",
                                                              "summary": summary, "episodes": episodes}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def experiment_tactics():
    from autobot.sim.env import ACTION_PROFILES
    return ACTION_PROFILES[experiment.ACTION_PROFILE]["tactics"]


def load_demonstrations():
    data = np.load(DATA_DIR / "demonstrations.npz")
    return {source: (data["observations"][data["source"] == index], data["labels"][data["source"] == index].astype(np.int64))
            for index, source in enumerate(SOURCES)}


# ---------------------------------------------------------------- fitting

def sample_batch(pools, balanced, generator):
    """Row indices per pool for one minibatch: an equal share of each pool, or uniform over all decisions."""
    import torch
    sizes = [len(labels) for _, labels in pools]
    if balanced:
        share = BATCH // len(pools)
        return [torch.randint(size, (share,), generator=generator) for size in sizes]
    flat = torch.randint(sum(sizes), (BATCH,), generator=generator); picks = []; low = 0
    for size in sizes:
        picks.append(flat[(flat >= low) & (flat < low + size)] - low); low += size
    return picks


def fit(model, pools, steps, balanced, seed):
    """Cross-entropy on the actor only. `pools` is {name: (observations, labels)}; returns loss and agreement per pool."""
    import torch
    policy = model.policy; policy.set_training_mode(True)
    optimiser = torch.optim.Adam(pilot.parameters_of(pilot.actor_modules(policy)), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(int(seed))
    tensors = [(torch.as_tensor(observations, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long)) for observations, labels in pools.values()]
    for _ in range(steps):
        picks = sample_batch(tensors, balanced, generator)
        inputs = torch.cat([observations[index] for (observations, _), index in zip(tensors, picks)])
        targets = torch.cat([labels[index] for (_, labels), index in zip(tensors, picks)])
        loss = torch.nn.functional.cross_entropy(imitation.logits(model, inputs), targets)
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    policy.set_training_mode(False); report = {}
    with torch.no_grad():
        for name, (observations, labels) in zip(pools, tensors):
            output = imitation.logits(model, observations)
            report[name] = {"decisions": int(len(labels)), "loss": float(torch.nn.functional.cross_entropy(output, labels)),
                            "agreement": float((output.argmax(1) == labels).float().mean())}
    return report


def _train_one(task):
    recipe, seed = task
    import torch
    torch.set_num_threads(1)
    from autobot.sim.ladder import GreedyActor
    out = run_directory(recipe, seed); out.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    data = load_demonstrations(); tactics = experiment_tactics(); ladder = experiment.load_ladder(); starts = load_starts()
    model = imitation.build_student(seed, None)
    untouched = pilot.flat(pilot.critic_modules(model.policy))
    pools = {name: ([data[name][0]], [data[name][1]]) for name in ("drill", "ordinary")}       # lists, so DAgger can append
    joined = lambda: {name: (np.concatenate(observations), np.concatenate(labels)) for name, (observations, labels) in pools.items()}
    held = lambda: {"heldout_agreement_on_ordinary_validation_demonstrations": imitation.agreement(model, *data["ordinary_validation"], tactics)}
    history = [{"stage": "behaviour_cloning", "gradient_steps": BC_STEPS, "fit": fit(model, joined(), BC_STEPS, recipe != "concat", seed), **held()}]
    if recipe == "balanced_dagger":
        env = experiment.make_env(); expert = diagnostic.FlankController(); expert.bind(env)
        plan = [("drill", start, DRILL_STEP_CAP) for start in drill_starts(ladder)] + [("ordinary", start, ORDINARY_STEP_CAP) for start in starts["train"]]
        for iteration in range(1, DAGGER_ITERATIONS + 1):
            driver = GreedyActor(model); added = Counter(); disagreed = Counter(); outcomes = Counter()
            for index, (source, start, cap) in enumerate(plan):
                observations, labels, taken, outcome = demonstrate(env, expert, start, DAGGER_SEED + 10_000 * seed + 1000 * iteration + index, driver, cap)
                pools[source][0].append(observations); pools[source][1].append(labels)
                added[source] += len(labels); disagreed[source] += int((taken != labels).sum()); outcomes[f"{source}:{outcome}"] += 1
            history.append({"stage": f"dagger_{iteration}", "gradient_steps": DAGGER_STEPS, "student_driven_decisions_added": dict(added),
                            "student_disagreed_while_driving": {source: disagreed[source] / max(added[source], 1) for source in added},
                            "student_driven_episode_outcomes": dict(sorted(outcomes.items())),
                            "fit": fit(model, joined(), DAGGER_STEPS, True, seed + iteration), **held()})
    if not np.array_equal(untouched, pilot.flat(pilot.critic_modules(model.policy))): raise RuntimeError("Fitting changed the value head")
    (out / "effective_config.json").write_text(json.dumps({
        "trainer": "scripts/combined_clone.py",
        "training": "SUPERVISED imitation of the scripted turn-then-advance controller on drill and ordinary starts. Not trained by PPO. "
                    "Simulation-only: the observation contains simulator truth.",
        "recipe": recipe, "seed": seed, "architecture": "mlp_64_64", "batch": BATCH, "learning_rate": LEARNING_RATE,
        "profile": experiment.PROFILE, "action_profile": experiment.ACTION_PROFILE, "previous_action_encoding": experiment.ENCODING,
        "reward_profile": "not used", "config_path": str(experiment.CONFIG.resolve()), "starts": experiment.relative(STARTS_PATH),
        "demonstrations": experiment.relative(DATA_DIR / "demonstrations.npz")}, indent=1), encoding="utf-8")
    model.save(out / "final_model")
    record = {"recipe": recipe, "seed": seed, "path": experiment.relative(out / "final_model.zip"), "history": history,
              "fitted_decisions": {name: int(sum(len(labels) for labels in pool[1])) for name, pool in pools.items()},
              "value_head_untouched": True, "wall_seconds": time.perf_counter() - started}
    (out / "training.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return f"{recipe}/seed{seed}", record


def train(args):
    tasks = [(recipe, seed) for recipe in args.recipes for seed in args.seeds]
    for recipe, seed in tasks:
        directory = run_directory(recipe, seed)
        if directory.exists() and any(directory.iterdir()): raise SystemExit(f"Refusing to overwrite {directory}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, record in pool.map(_train_one, tasks):
            last = record["history"][-1]
            print(f"{name:26s} {record['wall_seconds']:.0f}s  " + "  ".join(f"{pool_name} agreement {part['agreement']:.3f}" for pool_name, part in last["fit"].items())
                  + f"  validation demonstrations {last['heldout_agreement_on_ordinary_validation_demonstrations']['agreement']:.3f}", flush=True)


# ---------------------------------------------------------------- validation and the recipe choice

def policy_specs():
    specs = {f"{recipe}/seed{seed}": {"kind": "checkpoint", "path": str(run_directory(recipe, seed) / "final_model.zip"), "recipe": recipe, "seed": seed}
             for recipe in RECIPES for seed in SEEDS if (run_directory(recipe, seed) / "final_model.zip").is_file()}
    references = {"always_advance": {"kind": "always_advance"}, "turn_then_advance_controller": {"kind": "turn_then_advance_controller"}}
    references.update({f"drill_only_clone/seed{seed}": {"kind": "checkpoint", "path": str(pilot.clone_path(seed)), "recipe": "drill_only", "seed": seed}
                       for seed in SEEDS})
    return specs, references


def _validate_one(task):
    name, spec, cache = task
    if Path(cache).is_file(): return name, json.loads(Path(cache).read_text(encoding="utf-8"))
    ladder = experiment.load_ladder(); starts = load_starts(); actor, _ = pilot.load_actor(spec)
    sets = {"ladder_validation": ([start for rung in ladder["rungs"] for start in rung["validation"]], [VALIDATION_SEED]),
            "ordinary_validation": (starts["validation"], [VALIDATION_SEED])}
    result = {"episodes": pilot.run_sets(actor, experiment.make_env(), sets)}
    Path(cache).parent.mkdir(parents=True, exist_ok=True); Path(cache).write_text(json.dumps(result), encoding="utf-8")
    return name, result


def validation_table(episodes, first_gated):
    turn = [episode for episode in episodes["ladder_validation"] if pilot.needs_turn(episode, first_gated)]
    return {"turn_needed": {pit: pilot.cell([episode for episode in turn if episode["pit"] == pit]) for pit in pilot.PIT_NAMES},
            "ordinary": pilot.cell(episodes["ordinary_validation"])}


def pooled(tables_by_seed):
    """Sum the cells of several validation tables."""
    add = lambda cells: {key: sum(cell[key] for cell in cells) for key in cells[0]}
    return {"turn_needed": {pit: add([table["turn_needed"][pit] for table in tables_by_seed]) for pit in pilot.PIT_NAMES},
            "ordinary": add([table["ordinary"] for table in tables_by_seed])}


def select_recipe(pooled_tables):
    """Apply SELECTION to {recipe: pooled validation table}. Returns (recipe, per-recipe details)."""
    details = {}
    for recipe, table in pooled_tables.items():
        eligible = all(table["turn_needed"][pit]["wins"] >= SELECTION["turn_share"] * table["turn_needed"][pit]["episodes"] for pit in pilot.PIT_NAMES)
        details[recipe] = {"eligible": bool(eligible), "score": table["ordinary"]["wins"] - table["ordinary"]["own_falls"]}
    candidates = [recipe for recipe in RECIPES if recipe in details and details[recipe]["eligible"]] or [recipe for recipe in RECIPES if recipe in details]
    return max(candidates, key=lambda recipe: (details[recipe]["score"], -RECIPES.index(recipe))), details


def demo_seed(tables_by_seed):
    """The seed of the selected recipe to hand over: best validation score, then most turn-needed validation wins."""
    turn = lambda table: sum(table["turn_needed"][pit]["wins"] for pit in pilot.PIT_NAMES)
    return max(sorted(tables_by_seed), key=lambda seed: (tables_by_seed[seed]["ordinary"]["wins"] - tables_by_seed[seed]["ordinary"]["own_falls"], turn(tables_by_seed[seed])))


def validate(args):
    ladder = experiment.load_ladder(); first_gated = min(rung["abs_heading_offset_deg"] for rung in ladder["rungs"] if not rung["warmup"])
    specs, references = policy_specs()
    tasks = [(name, spec, str(DATA_DIR / "validation" / f"{name.replace('/', '_')}.json")) for name, spec in {**references, **specs}.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool: results = dict(pool.map(_validate_one, tasks))
    table = {name: validation_table(result["episodes"], first_gated) for name, result in results.items()}
    by_recipe = {recipe: {seed: table[f"{recipe}/seed{seed}"] for seed in SEEDS if f"{recipe}/seed{seed}" in table} for recipe in RECIPES}
    by_recipe = {recipe: seeds for recipe, seeds in by_recipe.items() if len(seeds) == len(SEEDS)}
    pooled_tables = {recipe: pooled(list(seeds.values())) for recipe, seeds in by_recipe.items()}
    selected, details = select_recipe(pooled_tables)
    output = {"purpose": "Choose the cloning recipe on validation starts, before any held-out start is run.",
              "rule": SELECTION, "validation_seed": VALIDATION_SEED,
              "starts": {"ladder_validation": "validation starts of config/heading_ladder_v1.json (those at 16 degrees or more count as turn-needed)",
                         "ordinary_validation": f"{experiment.relative(STARTS_PATH)} validation starts"},
              "per_policy": table, "pooled_over_seeds": pooled_tables, "selection": details, "selected_recipe": selected,
              "demo_seed": demo_seed(by_recipe[selected]),
              "training": {name: json.loads((run_directory(spec["recipe"], spec["seed"]) / "training.json").read_text(encoding="utf-8"))
                           for name, spec in specs.items()}}
    VALIDATION_PATH.write_text(json.dumps(output, indent=1), encoding="utf-8"); print(VALIDATION_PATH)
    show = lambda cell: f"{cell['wins']}/{cell['own_falls']}/{cell['timeouts']}"
    for name, row in table.items():
        print(f"{name:32s} turn TL {show(row['turn_needed']['top_left'])} BR {show(row['turn_needed']['bottom_right'])} of {row['turn_needed']['top_left']['episodes']} "
              f"| ordinary {show(row['ordinary'])} of {row['ordinary']['episodes']}  (wins/own falls/timeouts)")
    print("selection", details, "->", selected, "demo seed", output["demo_seed"])


# ---------------------------------------------------------------- held-out comparison

def keep_checks(candidate, drill_only, advance, controller):
    """Apply KEEP to one seed's held-out tables."""
    checks = {f"keeps_drill_{pit}": candidate["turn_needed"][pit]["wins"] >= KEEP["turn_share"] * drill_only["turn_needed"][pit]["wins"]
              and candidate["turn_needed"][pit]["wins"] > 0 for pit in pilot.PIT_NAMES}
    for name in ORDINARY_SETS:
        cell = candidate["ordinary"][name]
        checks[f"beats_advance_{name}"] = cell["wins"] > advance["ordinary"][name]["wins"]
        checks[f"half_of_controller_{name}"] = cell["wins"] >= KEEP["controller_share"] * controller["ordinary"][name]["wins"] and cell["wins"] > 0
        checks[f"few_own_falls_{name}"] = cell["own_falls"] <= KEEP["max_fall_share"] * cell["episodes"]
    return {"checks": {key: bool(value) for key, value in checks.items()}, "passes": bool(all(checks.values()))}


def heldout_tables(episodes, first_gated):
    table = pilot.tables(episodes, first_gated)
    table["ordinary"]["earlier_ordinary"] = table["ordinary"]["all"]          # the 20 ordinary starts of the earlier transfer test
    return table


def evaluate(args):
    ladder = experiment.load_ladder(); _, confirmation = pilot.load_mix()
    first_gated = min(rung["abs_heading_offset_deg"] for rung in ladder["rungs"] if not rung["warmup"])
    selection = json.loads(VALIDATION_PATH.read_text(encoding="utf-8")); selected = selection["selected_recipe"]
    specs, _ = policy_specs()
    tasks = [(name, spec, True, str(EVALUATION_DIR / f"{name.replace('/', '_')}.json")) for name, spec in specs.items()]
    # References were run on these starts and seeds in the pilot; the cache is reused, or rebuilt if a file is missing.
    tasks += [(name, {"kind": name}, True, str(REFERENCE_DIR / f"{name}.json")) for name in ("always_advance", "turn_then_advance_controller")]
    for ppo_seed, clone_seed in pilot.PAIRS.items():
        tasks.append((f"drill_only_clone/seed{clone_seed}", {"kind": "checkpoint", "path": str(pilot.clone_path(clone_seed))}, True,
                      str(REFERENCE_DIR / f"frozen_clone_seed{ppo_seed}.json")))
    with ProcessPoolExecutor(max_workers=args.workers) as pool: results = dict(pool.map(pilot._evaluate, tasks))
    expected = {name: sorted((start["name"], int(seed) + index) for seed in seeds for index, start in enumerate(starts))
                for name, (starts, seeds) in pilot.evaluation_sets(ladder, confirmation, True).items()}
    for name, result in results.items():
        got = {part: sorted((episode["scenario"], episode["seed"]) for episode in rows) for part, rows in result["episodes"].items()}
        if got != expected: raise SystemExit(f"{name} was not evaluated on the same held-out starts and seeds")
        if name.startswith("drill_only_clone") and result["path"] != experiment.relative(pilot.clone_path(int(name[-1]))):
            raise SystemExit(f"Cached reference {name} is not the drill-only clone")
    final = {name: heldout_tables(result["episodes"], first_gated) for name, result in results.items()}
    verdicts = {seed: keep_checks(final[f"{selected}/seed{seed}"], final[f"drill_only_clone/seed{seed}"], final["always_advance"],
                                  final["turn_then_advance_controller"]) for seed in SEEDS}
    passing = sum(verdict["passes"] for verdict in verdicts.values())
    output = {
        "purpose": "Can one supervised 64-64 clone hold both the turning drill and ordinary starts? No PPO; simulation-only.",
        "fixed": {"config": experiment.relative(experiment.CONFIG), "profile": experiment.PROFILE, "action_profile": experiment.ACTION_PROFILE,
                  "previous_action_encoding": experiment.ENCODING, "opponent": "scripted last_year", "environment": "AutoBotEnv, unchanged",
                  "evaluation": "greedy", "seeds": list(pilot.FINAL_SEEDS)},
        "teacher": "ordinary_start_diagnostic.FlankController: scripted turn-then-advance controller, reads simulator truth",
        "recipes": __doc__.split("Recipes (all start from the same fresh network for a seed):")[1].split("The recipe is chosen")[0].strip(),
        "fitting": {"batch": BATCH, "learning_rate": LEARNING_RATE, "behaviour_cloning_gradient_steps": BC_STEPS,
                    "dagger_iterations": DAGGER_ITERATIONS, "dagger_gradient_steps_each": DAGGER_STEPS,
                    "dagger_step_caps": {"drill": DRILL_STEP_CAP, "ordinary": ORDINARY_STEP_CAP}},
        "demonstrations": json.loads((DATA_DIR / "demonstrations.json").read_text(encoding="utf-8"))["summary"],
        "starts": {"demonstration_and_validation": experiment.relative(STARTS_PATH),
                   "held_out": {"turn_needed": f"ladder hold-out starts at {first_gated:g} degrees or more (108 starts x 3 seeds)",
                                "earlier_ordinary": "the 20 ordinary starts of the earlier transfer test (x 3 seeds)",
                                "new_confirmation": f"{len(confirmation)} ordinary confirmation starts ({experiment.relative(pilot.CONFIRMATION_PATH)}, x 3 seeds)"}},
        "selection": {key: selection[key] for key in ("rule", "pooled_over_seeds", "selection", "selected_recipe", "demo_seed")},
        "keep_rule": KEEP, "final": final, "selected_recipe": selected, "keep_verdict_by_seed": verdicts,
        "worth_keeping_as_a_simulator_demo": bool(passing >= 2), "demo_checkpoint": experiment.relative(run_directory(selected, selection["demo_seed"]) / "final_model.zip"),
        "results": results}
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8"); print(args.output)
    show = lambda cell: f"{cell['wins']}/{cell['own_falls']}/{cell['timeouts']}"
    for name, table in final.items():
        print(f"{name:32s} turn TL {show(table['turn_needed']['top_left'])} BR {show(table['turn_needed']['bottom_right'])} "
              f"| earlier ordinary {show(table['ordinary']['earlier_ordinary'])} | new confirmation {show(table['ordinary']['new_confirmation'])}")
    print("selected", selected, {seed: verdict["passes"] for seed, verdict in verdicts.items()}, "worth keeping:", output["worth_keeping_as_a_simulator_demo"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make")
    c = sub.add_parser("collect"); c.add_argument("--workers", type=int, default=12)
    t = sub.add_parser("train"); t.add_argument("--recipes", nargs="+", choices=RECIPES, default=list(RECIPES))
    t.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS)); t.add_argument("--workers", type=int, default=9)
    v = sub.add_parser("validate"); v.add_argument("--workers", type=int, default=14)
    e = sub.add_parser("evaluate"); e.add_argument("--workers", type=int, default=14)
    e.add_argument("--output", default=str(ROOT / "artifacts/combined-clone-comparison.json"))
    args = parser.parse_args()
    {"make": make, "collect": collect, "train": train, "validate": validate, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
