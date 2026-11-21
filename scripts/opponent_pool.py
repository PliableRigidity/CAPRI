"""Benchmark blue against a seeded pool of harder red driving styles. Simulation-only; red is scripted, not a human.

    python scripts/opponent_pool.py make                         # training / validation / final-test cases, written once
    python scripts/opponent_pool.py benchmark --split validation # run policies on a split (cached per policy)
    python scripts/opponent_pool.py styles                       # does each style drive differently? (applied commands)
    python scripts/opponent_pool.py decide                       # does the teacher earn a challenger?

Styles live in `autobot/sim/opponents.py`. A case is a start pose, a style, its resolved parameters and two seeds,
all stored in `config/opponent_pool_v1.json`, so any episode can be re-run exactly.

Splits are fixed before any data is collected:
    train       48 new starts x 4 styles, three of the four parameter combinations
    validation  16 new starts x (scripted + 4 styles), the same three combinations, new draws
    test        40 new starts x (scripted + 4 styles + pit_pusher). Half the cases use the parameter combination
                that training and validation never see, and pit_pusher appears nowhere else.

How outcomes are counted. A red fall is credited to blue only if the robots touched within the last half second
AND red does not also fall on the same case against a blue that never moves (Always-Stop). Everything else where
red falls is a red self-fall and is reported separately.
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
import combined_clone as combined  # noqa: E402
import heading_ladder as experiment  # noqa: E402
import ordinary_start_diagnostic as diagnostic  # noqa: E402

ROOT = experiment.ROOT
RUN = ROOT / "runs/overnight-2026-11-17"
POOL_PATH = ROOT / "config/opponent_pool_v1.json"
TEST_STARTS_PATH = ROOT / "config/opponent_pool_v1_test_starts.json"
REFERENCE = ROOT / "runs/combined-clone-balanced-dagger-seed2/final_model.zip"
POOL_SEED = 171117
STARTS_PER_FAMILY = {"train": 12, "validation": 4, "test": 10}
SEED_BASE = {"train": 300_000, "validation": 310_000, "test": 320_000}
FAMILIES = {"train_default": "default_spawn", "train_ordinary_left": "player_left_opponent_right",
            "train_ordinary_right": "player_right_opponent_left", "train_ordinary_diagonal": "opposed_diagonal"}
OUTCOMES = ("wins", "own_falls", "timeouts", "red_self_falls", "draws", "simulation_bugs")
BASELINES = {"always_stop": {"kind": "always_stop"}, "always_advance": {"kind": "always_advance"},
             "teacher": {"kind": "turn_then_advance_controller"}}
# Fixed before any benchmark was run.
TRAIN_RULE = {"data": "training and validation cases of the four harder styles (never the final test cases)",
              "viable_style": "the teacher's blue-caused wins exceed the reference clone's by at least 10% of the style's episodes, "
                              "and the teacher falls in on no more than 10% of them (or no more often than the reference clone)",
              "train_if": "at least two styles are viable; demonstrations are then collected on the viable styles only",
              "win_margin": 0.10, "max_fall_share": 0.10, "styles_needed": 2}


def styles_of(split):
    from autobot.sim.opponents import HELDOUT_STYLES, STYLES
    return {"train": STYLES, "validation": ("scripted",) + STYLES, "test": ("scripted",) + STYLES + HELDOUT_STYLES}[split]


def earlier_starts():
    """Every start used before tonight, held out or not: tonight's starts must all be new."""
    ladder = experiment.load_ladder(); mix, confirmation = pilot.load_mix(); starts = combined.load_starts()
    return pilot.heldout_starts(ladder) + confirmation + starts["train"] + starts["validation"] + mix["groups"]["ordinary"]


def build_pool(arena):
    from autobot.sim.opponents import SEEN_COMBOS, UNSEEN_COMBO, resolve_parameters
    from autobot.sim.scenarios import load_scenario_file
    archetypes = load_scenario_file(diagnostic.FROZEN / "training_curriculum.json")["groups"]["ordinary"]
    rng = np.random.default_rng(POOL_SEED); used = earlier_starts(); starts = {}
    for split, count in STARTS_PER_FAMILY.items():
        starts[split] = [dict(start, family=FAMILIES[start["derived_from"]]) for start in
                         pilot.jittered(archetypes, count, rng, f"pool_{split}", f"pool_{split}", used, arena)]
        used = used + starts[split]
    splits = {}
    for split in STARTS_PER_FAMILY:
        cases = []
        for style in styles_of(split):
            for index, start in enumerate(starts[split]):
                seed = SEED_BASE[split] + len(cases)
                case = {"name": f"{split}_{style}_{start['name']}", "split": split, "style": style, "family": start["family"], "seed": seed, "start": start}
                if style != "scripted":
                    # Test: every other start uses the combination that training and validation never see.
                    combo = UNSEEN_COMBO if split == "test" and index % 2 == 0 else SEEN_COMBOS[index % len(SEEN_COMBOS)]
                    case.update(combo=combo, driver_seed=seed + 500_000,
                                parameters=resolve_parameters(style, combo, np.random.default_rng([POOL_SEED, seed])))
                cases.append(case)
        splits[split] = cases
    pool = {"schema": "opponent_pool_v1", "seed": POOL_SEED, "config": experiment.relative(experiment.CONFIG),
            "styles": {"train": list(styles_of("train")), "validation": list(styles_of("validation")), "test": list(styles_of("test"))},
            "held_out_of_training_and_selection": {"starts": "every test start", "parameter_combination": UNSEEN_COMBO, "style": "pit_pusher"},
            "starts": f"seeded offsets (+-{pilot.JITTER_M} m, +-{pilot.JITTER_RAD} rad) of the four ordinary starts of the frozen curriculum; "
                      "none repeats a start used in any earlier experiment, and the three splits share none",
            "splits": splits}
    check_pool(pool)
    return pool


def check_pool(pool):
    """Splits share no start, repeat no earlier start, and only the test split has the unseen combination and pit_pusher."""
    from autobot.sim.opponents import HELDOUT_STYLES, UNSEEN_COMBO, combo_of
    unique = {split: list({case["start"]["name"]: case["start"] for case in cases}.values()) for split, cases in pool["splits"].items()}
    groups = list(unique.items()) + [("earlier", earlier_starts())]
    for position, (name, starts) in enumerate(groups[:-1]):
        for other_name, others in groups[position + 1:]:
            clashes = [(a["name"], b["name"]) for a in starts for b in others if pilot.same_case(a, b)]
            if clashes: raise ValueError(f"{name} starts repeat {other_name} starts: {clashes[:3]}")
    for split, cases in pool["splits"].items():
        for case in cases:
            if case["style"] == "scripted": continue
            if combo_of(case["style"], case["parameters"]) != case["combo"]: raise ValueError(f"{case['name']}: parameters are not in {case['combo']}")
            if split != "test" and (case["combo"] == UNSEEN_COMBO or case["style"] in HELDOUT_STYLES):
                raise ValueError(f"{case['name']}: held-out combination or style outside the test split")
    return pool


def load_pool():
    return check_pool(json.loads(POOL_PATH.read_text(encoding="utf-8")))


def make(args):
    from autobot.sim.core import load_config
    if POOL_PATH.exists(): raise SystemExit(f"{POOL_PATH} already exists; not overwriting.")
    pool = build_pool(load_config(str(experiment.CONFIG))["arena"])
    POOL_PATH.write_text(json.dumps(pool, indent=1), encoding="utf-8")
    flat = list({case["start"]["name"]: case["start"] for case in pool["splits"]["test"]}.values())
    TEST_STARTS_PATH.write_text(json.dumps(flat, indent=1), encoding="utf-8")
    print(POOL_PATH, {split: len(cases) for split, cases in pool["splits"].items()}); print(TEST_STARTS_PATH, len(flat))


# ---------------------------------------------------------------- running cases

def make_env():
    from autobot.sim.opponents import StyledOpponentEnv
    return StyledOpponentEnv(str(experiment.CONFIG), experiment.PROFILE, 0, opponent_model="last_year", opponent_mode="scripted",
                             reward_profile="outcome_only", action_profile=experiment.ACTION_PROFILE, previous_action_encoding=experiment.ENCODING)


def _run_chunk(task):
    spec, cases, keep = task
    from autobot.sim.opponents import case_rollout
    actor, _ = pilot.load_actor(spec); env = make_env()
    return [case_rollout(actor, env, case, keep_commands=keep) for case in cases]


def run_cases(spec, cases, workers, keep=False):
    """Episodes of one policy on a list of cases, in case order."""
    chunks = [cases[index::workers] for index in range(workers) if cases[index::workers]]
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool: parts = list(pool.map(_run_chunk, [(spec, chunk, keep) for chunk in chunks]))
    by_name = {episode["case"]: episode for part in parts for episode in part}
    return [by_name[case["name"]] for case in cases]


def benchmark_path(split, policy):
    return RUN / "benchmark" / f"{split}_{policy}.json"


def policy_specs(extra=()):
    specs = dict(BASELINES); specs["reference"] = {"kind": "checkpoint", "path": str(REFERENCE)}
    for item in extra:
        name, path = item.split("=", 1); specs[name] = {"kind": "checkpoint", "path": str((ROOT / path) if not Path(path).is_absolute() else Path(path))}
    return specs


def benchmark(args):
    pool = load_pool(); cases = pool["splits"][args.split]; specs = policy_specs(args.model)
    for name in args.policies or list(specs):
        path = benchmark_path(args.split, name)
        if path.is_file(): print(f"{name}: cached"); continue
        started = time.perf_counter(); episodes = run_cases(specs[name], cases, args.workers)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"policy": name, "spec": {key: (experiment.relative(value) if key == "path" else value) for key, value in specs[name].items()},
                                    "split": args.split, "episodes": episodes}), encoding="utf-8")
        print(f"{name}: {len(episodes)} episodes in {time.perf_counter() - started:.0f}s", flush=True)
    print_table(summary_of(args.split, args.policies or list(specs)))


def load_results(split, policies):
    return {name: json.loads(benchmark_path(split, name).read_text(encoding="utf-8"))["episodes"] for name in policies if benchmark_path(split, name).is_file()}


def credited(results):
    """Apply the matched Always-Stop rule: a win on a case where red also falls against a motionless blue is a red self-fall."""
    stop = {episode["case"]: episode["outcome"] for episode in results.get("always_stop", [])}
    adjusted = {}
    for name, episodes in results.items():
        rows = []
        for episode in episodes:
            falls_alone = stop.get(episode["case"]) in ("wins", "red_self_falls")
            rows.append(dict(episode, outcome="red_self_falls" if episode["outcome"] == "wins" and falls_alone else episode["outcome"],
                             red_also_falls_against_always_stop=bool(falls_alone)))
        adjusted[name] = rows
    return adjusted


def cell(episodes):
    counts = Counter(episode["outcome"] for episode in episodes)
    return {"episodes": len(episodes), **{key: counts[key] for key in OUTCOMES},
            "own_falls_while_in_contact": sum(episode["own_fall_while_in_contact"] for episode in episodes)}


def table(episodes):
    styles = list(dict.fromkeys(episode["style"] for episode in episodes)); families = sorted({episode["family"] for episode in episodes})
    harder = [episode for episode in episodes if episode["style"] != "scripted"]
    return {"by_style": {style: cell([e for e in episodes if e["style"] == style]) for style in styles},
            "by_style_and_family": {style: {family: cell([e for e in episodes if e["style"] == style and e["family"] == family]) for family in families}
                                    for style in styles},
            "by_family_harder_styles": {family: cell([e for e in harder if e["family"] == family]) for family in families},
            "by_combo_harder_styles": {combo: cell([e for e in harder if e.get("combo") == combo]) for combo in sorted({e.get("combo") for e in harder if e.get("combo")})},
            "harder_styles": cell(harder)}


def summary_of(split, policies):
    return {name: table(episodes) for name, episodes in credited(load_results(split, policies)).items()}


def print_table(summary):
    show = lambda c: f"{c['wins']}/{c['own_falls']}/{c['timeouts']}/{c['red_self_falls']}"
    print("wins / own falls / timeouts / red self-falls")
    for name, entry in summary.items():
        print(f"{name:16s} " + "  ".join(f"{style} {show(c)} of {c['episodes']}" for style, c in entry["by_style"].items()))


# ---------------------------------------------------------------- do the styles drive differently?

def styles(args):
    """Red's applied wheel commands by style, against a blue that never moves and against the reference clone."""
    results = {split: load_results(split, ["always_stop", "reference"]) for split in ("train", "validation")}
    keys = ("close_range_steps", "close_range_stopped_share", "close_range_driving_forward_share", "close_range_mean_forward_command",
            "mean_forward_command_when_driving_forward", "steering_while_moving_share", "turning_on_the_spot_share", "reversing_share",
            "stopped_share", "command_changes_per_second", "mean_seconds_between_command_changes")
    report = {}
    for policy in ("always_stop", "reference"):
        episodes = [episode for split in results.values() for episode in split.get(policy, [])]
        report[policy] = {}
        for style in dict.fromkeys(episode["style"] for episode in episodes):
            rows = [episode["red_driving"] for episode in episodes if episode["style"] == style]; entry = {"episodes": len(rows)}
            for key in keys:
                values = [row[key] for row in rows if row[key] is not None]
                entry[key] = None if not values else (int(sum(values)) if key == "close_range_steps" else round(float(np.mean(values)), 3))
            report[policy][style] = entry
    blue = report["reference"]
    checks = {
        "scripted_stops_at_close_range_and_persistent_follower_does_not":
            blue["scripted"]["close_range_stopped_share"] > 0.3 and blue["persistent_follower"]["close_range_stopped_share"] < 0.05
            and blue["persistent_follower"]["close_range_driving_forward_share"] > blue["scripted"]["close_range_driving_forward_share"] + 0.2,
        "persistent_follower_drives_faster_than_scripted":
            blue["persistent_follower"]["mean_forward_command_when_driving_forward"] > blue["scripted"]["mean_forward_command_when_driving_forward"] + 0.1,
        "curved_approach_steers_while_moving_most":
            blue["curved_approach"]["steering_while_moving_share"] > 0.4
            and blue["curved_approach"]["steering_while_moving_share"] > 2 * max(blue[s]["steering_while_moving_share"] for s in ("scripted", "persistent_follower", "stop_and_go")),
        "stop_and_go_pauses_and_backs_away":
            blue["stop_and_go"]["stopped_share"] > 0.08 and blue["stop_and_go"]["reversing_share"] > 0.03
            and blue["persistent_follower"]["reversing_share"] < 0.01 and blue["persistent_follower"]["stopped_share"] < 0.02,
        "reactive_changes_mixes_manoeuvres_and_holds_them":
            blue["reactive_changes"]["reversing_share"] > 0.01 and blue["reactive_changes"]["steering_while_moving_share"] > 0.1
            and blue["reactive_changes"]["mean_seconds_between_command_changes"] > 0.15}
    output = {"purpose": "Does each red style produce a different driving pattern? Measured from the wheel commands red actually applied.",
              "cases": "training and validation cases (no final test case)", "close_range_m": 0.22,
              "blue": {"always_stop": "blue never moves", "reference": "the reference clone"}, "patterns": report,
              "checks": {key: bool(value) for key, value in checks.items()}, "all_checks_pass": bool(all(checks.values()))}
    path = RUN / "style-patterns.json"; path.write_text(json.dumps(output, indent=1), encoding="utf-8"); print(path)
    for policy, rows in report.items():
        print(policy)
        for style, entry in rows.items(): print(f"  {style:20s} " + " ".join(f"{key.replace('_share', '').replace('close_range', 'close')}={entry[key]}" for key in keys))
    print(json.dumps(output["checks"], indent=1))


# ---------------------------------------------------------------- does the teacher earn a challenger?

def teacher_decision(summary_by_split):
    """Apply TRAIN_RULE to {split: summary} for the train and validation splits."""
    from autobot.sim.opponents import STYLES
    rows = {}
    for style in STYLES:
        total = lambda policy, key: sum(summary_by_split[split][policy]["by_style"][style][key] for split in summary_by_split)
        episodes = total("teacher", "episodes"); gain = total("teacher", "wins") - total("reference", "wins")
        fall_limit = max(TRAIN_RULE["max_fall_share"] * episodes, total("reference", "own_falls"))
        rows[style] = {"episodes": episodes, **{f"{policy}_{key}": total(policy, key) for policy in ("teacher", "reference") for key in ("wins", "own_falls", "timeouts", "red_self_falls")},
                       "teacher_win_gain": gain, "viable": bool(gain >= TRAIN_RULE["win_margin"] * episodes and total("teacher", "own_falls") <= fall_limit)}
    viable = [style for style, row in rows.items() if row["viable"]]
    return {"rule": TRAIN_RULE, "by_style": rows, "viable_styles": viable, "train": bool(len(viable) >= TRAIN_RULE["styles_needed"])}


def decide(args):
    policies = ["always_stop", "always_advance", "teacher", "reference"]
    summary = {split: summary_of(split, policies) for split in ("train", "validation")}
    decision = teacher_decision(summary); decision["benchmark"] = summary
    path = RUN / "teacher-decision.json"; path.write_text(json.dumps(decision, indent=1), encoding="utf-8"); print(path)
    for style, row in decision["by_style"].items():
        print(f"{style:20s} teacher {row['teacher_wins']}/{row['teacher_own_falls']}/{row['teacher_timeouts']}/{row['teacher_red_self_falls']}  "
              f"reference {row['reference_wins']}/{row['reference_own_falls']}/{row['reference_timeouts']}/{row['reference_red_self_falls']} of {row['episodes']}  "
              f"gain {row['teacher_win_gain']:+d}  viable {row['viable']}")
    print("viable styles:", decision["viable_styles"], "-> train a challenger:", decision["train"])


# ---------------------------------------------------------------- one file with the whole benchmark

def report(args):
    """Collect the benchmark on every split, the style check and the training decision into one file in the run directory."""
    policies = ["reference", "teacher", "always_advance", "always_stop"] + [item.split("=", 1)[0] for item in args.model]
    pool = load_pool(); test = summary_of("test", policies)
    reference = test["reference"]["by_style"]; harder = [style for style in reference if style != "scripted"]
    weakest = min(harder, key=lambda style: (reference[style]["wins"] - reference[style]["own_falls"], reference[style]["wins"]))
    read = lambda name: json.loads((RUN / name).read_text(encoding="utf-8")) if (RUN / name).is_file() else None
    decision = read("teacher-decision.json")
    output = {"purpose": "Blue checkpoints against a seeded pool of harder scripted red styles. Red is scripted throughout: none of this is evidence about a live human.",
              "run_directory": experiment.relative(RUN), "pool": experiment.relative(POOL_PATH), "config": experiment.relative(experiment.CONFIG),
              "policies": {"reference": experiment.relative(REFERENCE), "teacher": "scripted turn-then-advance controller (reads simulator truth)",
                           "always_advance": "blue drives straight ahead", "always_stop": "blue never moves"},
              "counting": {"wins": "red fell, the robots touched within the last half second, and red does not fall on this case against Always-Stop",
                           "red_self_falls": "red fell without that; reported separately and never credited to blue",
                           "own_falls": "blue fell; own_falls_while_in_contact says how many of those happened while touching red",
                           "timeouts": "nobody fell in 20 s"},
              "held_out": pool["held_out_of_training_and_selection"],
              "final_test": test, "weakest_harder_style_for_the_reference": weakest,
              "validation": summary_of("validation", policies), "train": summary_of("train", policies),
              "training_decision": decision and {key: decision[key] for key in ("rule", "by_style", "viable_styles", "train")},
              "challenger_trained": bool(args.model), "style_patterns": read("style-patterns.json"), "reference_recheck": read("reference-recheck.json")}
    path = RUN / "final-benchmark.json"; path.write_text(json.dumps(output, indent=1), encoding="utf-8"); print(path)
    print_table(test); print("weakest harder style for the reference:", weakest)


# ---------------------------------------------------------------- the recommended checkpoint, re-checked from disk

def recheck(args):
    """Reload the reference under its frozen settings and re-run it: final test cases, and its original held-out benchmark."""
    import hashlib
    from autobot.sim.cli import checkpoint_observation_metadata, checkpoint_ppo_settings, load_checkpoint
    model, encoding = load_checkpoint(REFERENCE, device="cpu"); metadata = checkpoint_observation_metadata(REFERENCE)
    env = make_env()
    report = {"checkpoint": experiment.relative(REFERENCE), "sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
              "reloads": {"previous_action_encoding": encoding, "metadata": metadata, "ppo_timesteps": checkpoint_ppo_settings(REFERENCE)["num_timesteps"],
                          "spaces_match_the_frozen_environment": bool(model.observation_space == env.observation_space and model.action_space == env.action_space),
                          "config": experiment.relative(experiment.CONFIG), "episode_seconds": env.sim.cfg["simulation"]["episode_seconds"]}}
    spec = policy_specs()["reference"]; cases = load_pool()["splits"]["test"]
    fresh = run_cases(spec, cases, args.workers); saved = load_results("test", ["reference"])["reference"]
    same = sum(a["outcome"] == b["outcome"] and a["steps"] == b["steps"] for a, b in zip(fresh, saved))
    report["final_test_cases_rerun"] = {"episodes": len(fresh), "identical_outcome_and_length": same}
    ladder = experiment.load_ladder(); _, confirmation = pilot.load_mix()
    first_gated = min(rung["abs_heading_offset_deg"] for rung in ladder["rungs"] if not rung["warmup"])
    _, original = pilot._evaluate(("reference", spec, True, None))
    earlier = json.loads((ROOT / "runs/combined-clone-evaluation/balanced_dagger_seed2.json").read_text(encoding="utf-8"))
    key = lambda episode: (episode["scenario"], episode["seed"])
    before = {part: {key(e): (e["outcome"], e["steps"]) for e in rows} for part, rows in earlier["episodes"].items()}
    matches = {part: sum(before[part].get(key(e)) == (e["outcome"], e["steps"]) for e in rows) for part, rows in original["episodes"].items()}
    tables = pilot.tables(original["episodes"], first_gated)
    report["original_benchmark_rerun"] = {"episodes": {part: len(rows) for part, rows in original["episodes"].items()},
                                          "identical_to_the_saved_evaluation": matches, "turn_needed": tables["turn_needed"], "ordinary": tables["ordinary"]}
    report["passed"] = bool(same == len(fresh) and all(matches[part] == len(rows) for part, rows in original["episodes"].items())
                            and encoding == experiment.ENCODING and report["reloads"]["spaces_match_the_frozen_environment"])
    path = RUN / "reference-recheck.json"; path.write_text(json.dumps(report, indent=1), encoding="utf-8"); print(json.dumps(report, indent=1))


# ---------------------------------------------------------------- replay of one case in the ordinary viewer

def replay_arguments(case, model, red_log, output_log):
    """`play` arguments that replay a recorded red: blue is recomputed live from the checkpoint, red follows the log."""
    index = [start["name"] for start in json.loads(TEST_STARTS_PATH.read_text(encoding="utf-8"))].index(case["start"]["name"])
    return ["--config", experiment.relative(experiment.CONFIG), "--seed", str(case["seed"]), "play", "--model", experiment.relative(model),
            "--profile", experiment.PROFILE, "--action-profile", experiment.ACTION_PROFILE, "--eval-scenarios", experiment.relative(TEST_STARTS_PATH),
            "--scenario-index", str(index), "--red-input-replay", experiment.relative(red_log), "--input-log", experiment.relative(output_log)]


def export_replay(args):
    """Write red's applied commands for one test case as a `play --red-input-replay` log, and check the replay reproduces it."""
    from autobot.sim import cli
    from autobot.sim.opponents import case_rollout
    case = next(case for case in load_pool()["splits"]["test"] if case["name"] == args.case)
    spec = policy_specs(args.model)[args.policy]; actor, _ = pilot.load_actor(spec)
    episode = case_rollout(actor, make_env(), case, keep_commands=True); hz = 50.0
    out = RUN / "replays"; out.mkdir(parents=True, exist_ok=True); red_log = out / f"{args.name}-red-commands.jsonl"
    lines = [{"type": "step", "time_s": (index + 1) / hz, "blue_tactic": tactic, "red_command": command, "source": f"scripted red style {case['style']}, not a human"}
             for index, (command, tactic) in enumerate(zip(episode["red_commands"], episode["blue_tactics"]))]
    red_log.write_text("".join(json.dumps(line) + chr(10) for line in lines), encoding="utf-8")
    arguments = replay_arguments(case, spec["path"], red_log, out / f"{args.name}-replayed.jsonl")
    captured = {}; original = cli.preview; cli.preview = lambda parsed: captured.setdefault("args", parsed)
    try: cli.main(arguments)
    finally: cli.preview = original
    session = cli.ReplaySession(captured["args"]); same = 0; done = False
    for line in lines:
        done, _ = session.step(tuple(line["red_command"])); same += session.current_tactic == line["blue_tactic"]
        if done: break
    record = {"case": case, "policy": args.policy, "episode": {key: value for key, value in episode.items() if key not in ("red_commands", "blue_tactics", "distances_m")},
              "red_log": experiment.relative(red_log), "command": "python -m autobot.sim.cli " + " ".join(arguments),
              "replay_check": {"steps": len(lines), "blue_tactics_reproduced": same, "end_reason": session.end_reason,
                               "simulated_seconds": round(float(session.sim.data.time), 2), "ended_on_the_last_logged_step": bool(done)}}
    (out / f"{args.name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("episode", "replay_check", "command")}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make")
    b = sub.add_parser("benchmark"); b.add_argument("--split", choices=list(STARTS_PER_FAMILY), required=True)
    b.add_argument("--policies", nargs="+"); b.add_argument("--model", action="append", default=[], metavar="NAME=CHECKPOINT")
    b.add_argument("--workers", type=int, default=14)
    sub.add_parser("styles"); sub.add_parser("decide")
    r = sub.add_parser("export-replay"); r.add_argument("--case", required=True); r.add_argument("--policy", default="reference")
    r.add_argument("--name", required=True); r.add_argument("--model", action="append", default=[], metavar="NAME=CHECKPOINT")
    c = sub.add_parser("recheck"); c.add_argument("--workers", type=int, default=12)
    f = sub.add_parser("report"); f.add_argument("--model", action="append", default=[], metavar="NAME=CHECKPOINT")
    args = parser.parse_args()
    {"make": make, "benchmark": benchmark, "styles": styles, "decide": decide, "export-replay": export_replay, "recheck": recheck,
     "report": report}[args.command](args)


if __name__ == "__main__":
    main()
