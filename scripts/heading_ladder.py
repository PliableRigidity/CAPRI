"""Heading ladder experiment: can PPO learn turn-then-push when every episode is spent on it? Simulation-only.

    python scripts/heading_ladder.py scan        # where does Always-Advance stop winning?
    python scripts/heading_ladder.py make        # build the ladder; refuses rungs that do not separate the references
    python scripts/heading_ladder.py train --seed 23 --output runs/heading-ladder-seed23
    python scripts/heading_ladder.py evaluate --runs runs/heading-ladder-seed23 ...

Fixed for every run: frozen 20 s config, sim_debug observations, experimental5_v1 (five tactics),
masked_zero_v1 previous action, outcome_only reward, scripted opponent, gamma 0.99, the PPO settings of
`autobot.sim.cli train`. The only thing the ladder changes is which start pose a training episode gets.
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
import ordinary_start_diagnostic as diagnostic  # noqa: E402

ROOT = diagnostic.ROOT
CONFIG = diagnostic.CONFIG
LADDER_PATH = ROOT / "config/heading_ladder_v1.json"
PROFILE = "sim_debug"; ACTION_PROFILE = "experimental5_v1"; ENCODING = "masked_zero_v1"; GAMMA = 0.99
VALIDATION_SEEDS = (60017, 61017)          # gate: every validation start once per seed
FINAL_SEEDS = (80017, 81017, 82017)        # final check: seeds never used for gating
SCAN_SEED = 70117
SEPARATION = {"advance_max_win_rate": 0.25, "controller_min_win_rate": 0.90}
REFERENCES = ("always_advance", "turn_then_advance_controller")


class RandomTactic:
    """Uniform random tactic every decision: what undirected exploration gets for free."""
    def __init__(self, seed): self.rng = np.random.default_rng(seed)

    def predict(self, observation, deterministic=True): return np.array(int(self.rng.integers(5))), None


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def make_env(seed=0):
    from autobot.sim.env import AutoBotEnv
    return AutoBotEnv(str(CONFIG), PROFILE, seed, opponent_model="last_year", opponent_mode="scripted",
                      reward_profile="outcome_only", action_profile=ACTION_PROFILE, previous_action_encoding=ENCODING)


def reference_policy(kind, seed=0):
    from autobot.sim.env import ScriptedBaseline, StopBaseline
    return {"always_advance": ScriptedBaseline, "always_stop": StopBaseline, "turn_then_advance_controller": diagnostic.FlankController,
            "random": lambda: RandomTactic(seed)}[kind]()


def load_ladder(path=LADDER_PATH):
    from autobot.sim.ladder import validate_ladder
    return validate_ladder(json.loads(Path(path).read_text(encoding="utf-8")))


def _reference_chunk(task):
    kind, starts, seeds = task
    from autobot.sim.ladder import ladder_rollout
    env = make_env(); episodes = []
    for seed in seeds:
        for index, start in enumerate(starts):
            episodes.append(ladder_rollout(reference_policy(kind, seed + index), env, start, seed + index))
    return kind, episodes


def reference_rollouts(kinds, starts, seeds, workers):
    """Reference policies on a list of starts, spread over processes. Returns {kind: [episodes]}."""
    chunks = [starts[index::workers] for index in range(workers)]
    tasks = [(kind, chunk, [seed + 1000 * offset for seed in seeds]) for kind in kinds for offset, chunk in enumerate(chunks) if chunk]
    results = {kind: [] for kind in kinds}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for kind, episodes in pool.map(_reference_chunk, tasks): results[kind] += episodes
    return results


def scan(args):
    """Advance / controller / random win rates against heading offset and distance, to place the first rung."""
    from autobot.sim.ladder import PITS, ladder_start
    starts = [ladder_start(pit, distance, sign * offset, "scan") for offset in args.offsets for distance in args.distances
              for pit in PITS for sign in (1, -1)]
    results = reference_rollouts(list(REFERENCES), starts, [SCAN_SEED], args.workers)
    results.update(reference_rollouts(["random"], starts, [SCAN_SEED + 10_000 * repeat for repeat in range(args.random_repeats)], args.workers))
    table = {}
    for kind, episodes in results.items():
        for episode in episodes:
            cell = table.setdefault(kind, {}).setdefault(f"{abs(episode['heading_offset_deg']):g}", {}).setdefault(f"{episode['distance_m']:.2f}", Counter())
            cell[episode["outcome"]] += 1; cell["episodes"] += 1
    output = {"purpose": "place the first ladder rung: the smallest heading offset where a plain advance stops winning "
                         "while the turn-then-advance controller still wins",
              "config": relative(CONFIG), "offsets_deg": args.offsets, "distances_m": args.distances,
              "note": "advance and controller are deterministic per start (4 starts per cell: 2 pits x 2 directions); "
                      f"random is {args.random_repeats} sampled episodes per start",
              "wins_by_abs_offset_and_distance": {kind: {offset: {distance: f"{cell['wins']}/{cell['episodes']}" for distance, cell in sorted(row.items())}
                                                         for offset, row in rows.items()} for kind, rows in table.items()},
              "own_falls_by_abs_offset": {kind: {offset: sum(cell["own_falls"] for cell in row.values()) for offset, row in rows.items()}
                                          for kind, rows in table.items()},
              "episodes": results}
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(args.output)
    for kind, rows in output["wins_by_abs_offset_and_distance"].items():
        print(kind)
        for offset, row in rows.items(): print(f"  {offset:>5} deg  " + "  ".join(f"d{distance} {value}" for distance, value in row.items()))


def make(args):
    """Build the ladder and verify, before any training, that each gated rung separates the two references."""
    from autobot.sim.ladder import build_ladder, summarise_episodes
    ladder = build_ladder(args.rungs, warmup_offset_deg=args.warmup, holdout_offsets_deg=args.holdout_offsets, seed=args.ladder_seed,
                          pits=args.pits)
    problems = []
    for rung in ladder["rungs"]:
        reference = {}
        for split in ("validation", "train"):
            results = reference_rollouts(list(REFERENCES), rung[split], list(VALIDATION_SEEDS), args.workers)
            reference[split] = {kind: summarise_episodes(episodes) for kind, episodes in results.items()}
        chance = reference_rollouts(["random"], rung["train"], [SCAN_SEED + 10_000 * repeat for repeat in range(3)], args.workers)
        reference["train"]["random"] = summarise_episodes(chance["random"])
        rates = {(split, kind): reference[split][kind]["wins"] / reference[split][kind]["episodes"]
                 for split in ("validation", "train") for kind in REFERENCES}
        if rung["warmup"]:
            ok = all(rate >= SEPARATION["controller_min_win_rate"] for rate in rates.values())
            rule = "warm-up: a plain advance and the controller both win (no turn needed)"
        else:
            ok = all(rates[split, "always_advance"] <= SEPARATION["advance_max_win_rate"]
                     and rates[split, "turn_then_advance_controller"] >= SEPARATION["controller_min_win_rate"] for split in ("validation", "train"))
            rule = "gated: a plain advance fails and the turn-then-advance controller wins"
        rung["reference"] = reference; rung["separation_rule"] = rule; rung["separation_verified"] = bool(ok)
        line = "  ".join(f"{split} adv {reference[split]['always_advance']['wins']}/{reference[split]['always_advance']['episodes']} "
                         f"ctrl {reference[split]['turn_then_advance_controller']['wins']}/{reference[split]['turn_then_advance_controller']['episodes']}"
                         for split in ("validation", "train"))
        label = ", warm-up" if rung["warmup"] else ""
        print(f"rung {rung['index']} ({rung['abs_heading_offset_deg']:g} deg{label}): {line}  "
              f"random train {reference['train']['random']['wins']}/{reference['train']['random']['episodes']}  "
              f"{'OK' if ok else 'DOES NOT SEPARATE'}", flush=True)
        if not ok: problems.append(rung["index"])
    if problems and not args.report_only:
        raise SystemExit(f"Rungs {problems} do not separate Always-Advance from the controller; ladder not written. Adjust the offsets.")
    if args.report_only: return
    ladder["reference_seeds"] = list(VALIDATION_SEEDS); ladder["separation"] = SEPARATION; ladder["config"] = relative(CONFIG)
    Path(args.output).write_text(json.dumps(ladder, indent=1), encoding="utf-8")
    # Flat list so `autobot.sim.cli replay --eval-scenarios ... --scenario-index N` can show any validation or hold-out start.
    flat = [start for rung in ladder["rungs"] for start in rung["validation"]] + ladder["holdout"]
    scenarios_path = Path(args.output).with_name(Path(args.output).stem + "_scenarios.json")
    scenarios_path.write_text(json.dumps(flat, indent=1), encoding="utf-8")
    print(args.output, f"rungs {len(ladder['rungs'])}, train starts {sum(len(r['train']) for r in ladder['rungs'])}, "
          f"validation {sum(len(r['validation']) for r in ladder['rungs'])}, holdout {len(ladder['holdout'])}")
    print(scenarios_path, len(flat))


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from autobot.sim import cli
    from autobot.sim.core import load_config
    from autobot.sim.env import ACTION_PROFILES
    from autobot.sim.ladder import TURN_TACTICS, LadderEnv, RungGate, gate_passes, ladder_rollout, run_starts, summarise_episodes
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): raise SystemExit(f"Refusing to overwrite non-empty run directory: {out}")
    ladder = load_ladder(args.ladder)
    if not all(rung.get("separation_verified") for rung in ladder["rungs"]):
        raise SystemExit("Ladder has rungs whose separation was not verified; rebuild it with `make`.")
    if args.torch_threads < 1: raise SystemExit("--torch-threads must be at least 1")
    torch.set_num_threads(args.torch_threads); out.mkdir(parents=True, exist_ok=True)
    tactics = ACTION_PROFILES[ACTION_PROFILE]["tactics"]
    env = make_vec_env(lambda: LadderEnv(ladder, str(CONFIG), PROFILE, opponent_model="last_year", opponent_mode="scripted",
                                         reward_profile="outcome_only", action_profile=ACTION_PROFILE,
                                         previous_action_encoding=ENCODING, ladder_stream=args.seed), n_envs=1, seed=args.seed)
    # Same PPO construction as `autobot.sim.cli train` (MlpPolicy, n_steps 256, batch 32, library defaults otherwise).
    model = PPO("MlpPolicy", env, device="cpu", seed=args.seed, verbose=0, n_steps=256, batch_size=32, gamma=GAMMA)
    actual = cli.ppo_hyperparameters(model)
    if actual["gamma"] != GAMMA or actual["rollout_buffer_gamma"] != GAMMA: raise SystemExit(f"PPO is not using gamma={GAMMA}: {actual}")
    setattr(model, cli.OBSERVATION_METADATA_ATTRIBUTE, cli.observation_metadata(PROFILE, ACTION_PROFILE, ENCODING))
    gate_settings = {"validation_seeds": list(VALIDATION_SEEDS), "min_win_rate": args.min_win_rate, "min_margin_over_advance": args.min_margin,
                     "consecutive_validations": args.consecutive, "eval_freq": args.eval_freq, "rung_budget_timesteps": args.rung_budget,
                     "rule": "greedy policy on the frontier rung's validation starts; cleared after consecutive passes; "
                             "a rung not cleared within the budget stops the run"}
    metadata = {"trainer": "scripts/heading_ladder.py", "config": load_config(str(CONFIG)), "config_path": str(CONFIG.resolve()),
                "seed": args.seed, "profile": PROFILE, "n_envs": 1, "opponent_mode": "scripted", "opponent_chassis": "last_year",
                "reward_profile": "outcome_only", "action_profile": ACTION_PROFILE, "previous_action_encoding": ENCODING, "gamma": GAMMA,
                "action_names": list(tactics), "observation_normalization": False,
                "observation_fields": cli.observation_fields(PROFILE, ACTION_PROFILE, ENCODING), "training_curriculum": None,
                "ladder_path": str(Path(args.ladder).resolve()), "ladder": ladder, "gate": gate_settings,
                "timesteps_cap": args.timesteps, "ppo_hyperparameters": actual}
    (out / "effective_config.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    metrics_path = out / "metrics.jsonl"

    def log(record):
        with metrics_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
    log({"type": "run_start", "previous_action_encoding": ENCODING, "action_profile": ACTION_PROFILE, "profile": PROFILE,
         "ppo_hyperparameters": actual, "gate": gate_settings})

    class LadderCallback(BaseCallback):
        def __init__(self):
            super().__init__(); self.gate = RungGate(args.consecutive); self.rung = 0; self.rung_started = 0
            self.eval_env = make_env(); self.counts = Counter(); self.first_contact = None; self.pre_turns = 0
            self.episode_index = 0; self.cleared = []; self.stop_reason = None
            self.next_eval = args.eval_freq; self.next_checkpoint = args.checkpoint_freq; self.started = time.perf_counter()

        def profile(self, label):
            """Greedy policy on every rung's validation starts (one seed), to see the whole ladder at this moment."""
            rows = []
            for rung in ladder["rungs"]:
                summary = summarise_episodes(run_starts(self.model, self.eval_env, rung["validation"], VALIDATION_SEEDS[:1]))
                rows.append({"rung": rung["index"], "abs_heading_offset_deg": rung["abs_heading_offset_deg"], **summary})
            log({"type": "ladder_profile", "timesteps": self.num_timesteps, "at": label, "frontier": self.rung, "rungs": rows})

        def validate(self):
            rung = ladder["rungs"][self.rung]; advance = rung["reference"]["validation"]["always_advance"]
            planned = [(start, seed + index) for seed in VALIDATION_SEEDS for index, start in enumerate(rung["validation"])]
            assert advance["episodes"] == len(planned), "Always-Advance reference was not run on the same validation episodes"
            episodes = []
            for start, seed in planned:
                episodes.append(ladder_rollout(self.model, self.eval_env, start, seed))
                wins = sum(episode["outcome"] == "wins" for episode in episodes)
                # Failed validations are mostly 20 s timeouts: stop once the remaining episodes cannot make it pass.
                if not gate_passes(wins + len(planned) - len(episodes), advance["wins"], len(planned), min_win_rate=args.min_win_rate,
                                   min_margin=args.min_margin, warmup=rung["warmup"]): break
            summary = summarise_episodes(episodes)
            passed = len(episodes) == len(planned) and gate_passes(summary["wins"], advance["wins"], len(planned),
                                                                   min_win_rate=args.min_win_rate, min_margin=args.min_margin, warmup=rung["warmup"])
            cleared = self.gate.update(passed)
            log({"type": "validation", "timesteps": self.num_timesteps, "rung": self.rung,
                 "abs_heading_offset_deg": rung["abs_heading_offset_deg"], "warmup": rung["warmup"], "policy": summary,
                 "episodes_planned": len(planned), "stopped_early_as_failed": len(episodes) < len(planned),
                 "always_advance_wins": advance["wins"], "passed": passed,
                 "consecutive_passes": self.gate.required if cleared else self.gate.streak,
                 "timesteps_on_rung": self.num_timesteps - self.rung_started, "elapsed_seconds": time.perf_counter() - self.started})
            print(f"[{self.num_timesteps:>8}] rung {self.rung} ({rung['abs_heading_offset_deg']:g} deg) validation "
                  f"{summary['wins']}/{summary['episodes']} wins ({summary['wins_with_initial_turn']} with an initial turn), "
                  f"{summary['own_falls']} own falls; advance {advance['wins']}; {'pass' if passed else 'fail'}", flush=True)
            if cleared:
                self.cleared.append(self.rung); self.model.save(out / f"rung_{self.rung}_cleared")
                log({"type": "rung_cleared", "timesteps": self.num_timesteps, "rung": self.rung, "warmup": rung["warmup"],
                     "abs_heading_offset_deg": rung["abs_heading_offset_deg"], "timesteps_on_rung": self.num_timesteps - self.rung_started,
                     "episode_index": self.episode_index})
                self.profile(f"rung_{self.rung}_cleared")
                if self.rung + 1 == len(ladder["rungs"]): self.stop_reason = "ladder_complete"
                else:
                    self.rung += 1; self.rung_started = self.num_timesteps
                    self.training_env.env_method("set_rung", self.rung)
            elif self.num_timesteps - self.rung_started >= args.rung_budget:
                self.stop_reason = "rung_stalled"
                log({"type": "rung_stalled", "timesteps": self.num_timesteps, "rung": self.rung, "warmup": rung["warmup"],
                     "abs_heading_offset_deg": rung["abs_heading_offset_deg"], "timesteps_on_rung": self.num_timesteps - self.rung_started})

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
                     "frontier": info.get("curriculum_stage"), "sampled_rung": info.get("scenario_group"), "scenario": info.get("scenario"),
                     "outcome": outcome, "steps": sum(self.counts.values()), "first_contact_step": self.first_contact,
                     "turn_decisions_before_contact": self.pre_turns, "actions": dict(self.counts)})
                self.counts = Counter(); self.first_contact = None; self.pre_turns = 0; self.episode_index += 1
            if self.num_timesteps >= self.next_checkpoint:
                self.next_checkpoint += args.checkpoint_freq
                (out / "checkpoints").mkdir(exist_ok=True)
                self.model.save(out / "checkpoints" / f"rl_model_{self.num_timesteps}_steps"); self.model.save(out / "latest_model")
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_freq; self.validate()
            return self.stop_reason is None

    callback = LadderCallback(); started = time.perf_counter()
    model.learn(args.timesteps, callback=callback)
    elapsed = time.perf_counter() - started; model.save(out / "latest_model"); model.save(out / "final_model")
    gated = [index for index in callback.cleared if not ladder["rungs"][index]["warmup"]]
    summary = {"type": "run_complete", "stop_reason": callback.stop_reason or "timestep_cap", "ending_timesteps": int(model.num_timesteps),
               "frontier_rung": callback.rung, "rungs_cleared": callback.cleared, "gated_rungs_cleared": gated,
               "largest_cleared_offset_deg": max((ladder["rungs"][index]["abs_heading_offset_deg"] for index in gated), default=None),
               "episodes": callback.episode_index, "elapsed_seconds": elapsed, "environment_steps_per_second": model.num_timesteps / elapsed}
    log(summary); print(json.dumps(summary, indent=1))


def transfer_scenarios():
    """Ordinary and normal starts: a separate transfer test, never part of the ladder."""
    from autobot.sim.scenarios import load_scenario_file
    fresh = load_scenario_file(ROOT / "config/normal_start_confirmation_scenarios.json")
    return diagnostic.ordinary_scenarios() + [dict(item, group="fresh_normal") for item in fresh]


def _evaluate_policy(task):
    name, spec, ladder, transfer = task
    import torch
    torch.set_num_threads(1)
    from autobot.sim.cli import checkpoint_observation_metadata, checkpoint_ppo_settings, load_checkpoint
    from autobot.sim.ladder import GreedyActor, run_starts
    result = {"kind": spec["kind"]}
    if spec["kind"] == "checkpoint":
        model, encoding = load_checkpoint(spec["path"], device="cpu"); metadata = checkpoint_observation_metadata(spec["path"])
        settings = checkpoint_ppo_settings(spec["path"])
        assert encoding == ENCODING and metadata["action_profile"] == ACTION_PROFILE and metadata["observation_profile"] == PROFILE, spec["path"]
        result.update(path=relative(spec["path"]), gamma=settings["gamma"], num_timesteps=settings["num_timesteps"],
                      previous_action_encoding=encoding, action_profile=metadata["action_profile"])
        model = GreedyActor(model)     # same greedy choices, without the per-decision tensor overhead
    else: model = reference_policy(spec["kind"])
    env = make_env()
    result["validation"] = [episode for rung in ladder["rungs"] for episode in run_starts(model, env, rung["validation"], FINAL_SEEDS[:1])]
    result["holdout"] = run_starts(model, env, ladder["holdout"], FINAL_SEEDS)
    result["transfer"] = run_starts(model, env, transfer, FINAL_SEEDS)
    return name, result


def by_offset(episodes):
    from autobot.sim.ladder import summarise_episodes
    table = {}
    for episode in episodes: table.setdefault(f"{abs(episode['heading_offset_deg']):g}", []).append(episode)
    return {offset: summarise_episodes(rows) for offset, rows in sorted(table.items(), key=lambda item: float(item[0]))}


def run_record(run):
    """What a training run logged: rungs cleared, where it stopped, training outcomes per sampled rung."""
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    complete = next((row for row in rows if row["type"] == "run_complete"), None)
    episodes = [row for row in rows if row["type"] == "train_episode"]; training = {}
    tactic_counts = {}
    for row in episodes:
        cell = training.setdefault(row["sampled_rung"], Counter()); cell[row["outcome"]] += 1; cell["episodes"] += 1
        cell["episodes_with_a_turn_before_contact"] += row["turn_decisions_before_contact"] > 0
        if row["outcome"] == "win" and row["turn_decisions_before_contact"] > 0: cell["wins_with_a_turn_before_contact"] += 1
        tactic_counts.setdefault(row["sampled_rung"], Counter()).update(row["actions"])
    for name, counts in tactic_counts.items():
        # Share of sampled (exploring) decisions per tactic: has the policy stopped trying turns?
        training[name] = dict(training[name], tactic_share={tactic: round(count / sum(counts.values()), 4) for tactic, count in sorted(counts.items())})
    return {"complete": complete, "rungs_cleared": [row for row in rows if row["type"] == "rung_cleared"],
            "stalled": next((row for row in rows if row["type"] == "rung_stalled"), None),
            "validations": [{key: row[key] for key in ("timesteps", "rung", "passed", "always_advance_wins", "episodes_planned", "stopped_early_as_failed")}
                            | {key: row["policy"][key] for key in ("episodes", "wins", "own_falls", "timeouts", "wins_with_initial_turn")}
                            for row in rows if row["type"] == "validation"],
            "ladder_profiles": [row for row in rows if row["type"] == "ladder_profile"],
            "training_outcomes_by_sampled_rung": {name: dict(cell) for name, cell in sorted(training.items())},
            "training_episodes": len(episodes)}


def evaluate(args):
    from autobot.sim.ladder import summarise_episodes
    ladder = load_ladder(args.ladder); transfer = transfer_scenarios()
    specs = {"always_stop": {"kind": "always_stop"}, "always_advance": {"kind": "always_advance"},
             "turn_then_advance_controller": {"kind": "turn_then_advance_controller"}}
    runs = {}
    for item in args.runs:
        run = (ROOT / item) if not Path(item).is_absolute() else Path(item); runs[run.name] = run_record(run)
        for path in sorted(run.glob("rung_*_cleared.zip")) + [run / "final_model.zip"]:
            if path.is_file(): specs[f"{run.name}/{path.stem}"] = {"kind": "checkpoint", "path": str(path)}
    for item in args.extra:
        name, path = item.split("=", 1); specs[name] = {"kind": "checkpoint", "path": str((ROOT / path) if not Path(path).is_absolute() else Path(path))}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = dict(pool.map(_evaluate_policy, [(name, spec, ladder, transfer) for name, spec in specs.items()]))
    first_gated = min(rung["abs_heading_offset_deg"] for rung in ladder["rungs"] if not rung["warmup"])
    needs_turn = lambda episode: abs(episode["heading_offset_deg"]) >= first_gated
    summary = {}
    for name, result in results.items():
        holdout = result["holdout"]; groups = {}
        for episode in result["transfer"]: groups.setdefault(episode["group"], []).append(episode)
        summary[name] = {
            "validation_by_abs_offset_deg": by_offset(result["validation"]),
            "validation_by_pit_and_abs_offset_deg": {pit: by_offset([e for e in result["validation"] if e["pit"] == pit])
                                                     for pit in ("top_left", "bottom_right")},
            "holdout_by_abs_offset_deg": by_offset(holdout),
            "holdout_needing_a_turn": summarise_episodes([e for e in holdout if needs_turn(e)]),
            "holdout_needing_a_turn_by_kind": {kind: summarise_episodes([e for e in holdout if e["holdout_kind"] == kind and needs_turn(e)])
                                               for kind in ("combination", "level")},
            "holdout_needing_a_turn_by_pit": {pit: summarise_episodes([e for e in holdout if e["pit"] == pit and needs_turn(e)])
                                              for pit in ("top_left", "bottom_right")},
            "holdout_needing_a_turn_by_distance_m": {
                f"{distance:.2f}": summarise_episodes([e for e in holdout if e["distance_m"] == distance and needs_turn(e)])
                for distance in sorted({e["distance_m"] for e in holdout})},
            "transfer": {group: summarise_episodes(rows) for group, rows in groups.items()}}
    output = {"purpose": "Heading ladder: can PPO learn turn-then-push when every training episode is a misaligned near-pit start?",
              "fixed": {"config": relative(CONFIG), "profile": PROFILE, "action_profile": ACTION_PROFILE,
                        "previous_action_encoding": ENCODING, "gamma": GAMMA, "reward_profile": "outcome_only", "opponent": "scripted last_year"},
              "ladder": relative(args.ladder),
              "rungs": [{key: rung[key] for key in ("index", "abs_heading_offset_deg", "warmup", "reference")} for rung in ladder["rungs"]],
              "definitions": {
                  "validation": "per-rung starts used to gate the ladder during training (so not independent evidence); one seed here",
                  "holdout": "never used for training or gating: combinations left out of training on both pits, plus offsets between the rungs "
                             f"at distances {ladder['holdout_distances_m']} m; {len(FINAL_SEEDS)} seeds",
                  "holdout_needing_a_turn": f"hold-out starts at offsets of {first_gated:g} deg or more, which a plain advance does not win",
                  "transfer": "ordinary training starts, frozen normal starts and fresh normal starts; kept separate from the ladder",
                  "initial_turn": "before first touching red, blue rotated towards the push line by at least half the starting offset",
                  "seeds": {"validation_gate": list(VALIDATION_SEEDS), "final": list(FINAL_SEEDS)}},
              "runs": runs, "summary": summary, "results": results}
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(args.output)
    for name, entry in summary.items():
        turn = entry["holdout_needing_a_turn"]; normal = {group: f"{cell['wins']}/{cell['episodes']}" for group, cell in entry["transfer"].items()}
        print(f"{name:48s} holdout needing a turn {turn['wins']}/{turn['episodes']} wins ({turn['wins_with_initial_turn']} initial turn), "
              f"{turn['own_falls']} falls | transfer {normal}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan"); s.add_argument("--offsets", type=float, nargs="+", default=[6, 8, 10, 12, 14, 16, 18, 20, 24])
    s.add_argument("--distances", type=float, nargs="+", default=[0.34, 0.40, 0.46, 0.52, 0.58])
    s.add_argument("--random-repeats", type=int, default=5); s.add_argument("--workers", type=int, default=12)
    s.add_argument("--output", default=str(ROOT / "artifacts/heading-ladder-first-rung-scan.json"))
    m = sub.add_parser("make"); m.add_argument("--rungs", type=float, nargs="+", default=[16, 24, 34, 45, 60, 90])
    m.add_argument("--warmup", type=float, default=10.0, help="ungated first rung that a plain advance already wins")
    m.add_argument("--holdout-offsets", type=float, nargs="+", default=[20, 29, 39, 52, 75, 120])
    m.add_argument("--ladder-seed", type=int, default=7); m.add_argument("--workers", type=int, default=12)
    m.add_argument("--pits", nargs="+", choices=["top_left", "bottom_right"], help="train and validate on these pits only (default: both)")
    m.add_argument("--report-only", action="store_true", help="print the separation table without writing the ladder")
    m.add_argument("--output", default=str(LADDER_PATH))
    t = sub.add_parser("train"); t.add_argument("--seed", type=int, required=True); t.add_argument("--output", required=True)
    t.add_argument("--ladder", default=str(LADDER_PATH)); t.add_argument("--timesteps", type=int, default=1_500_000)
    t.add_argument("--rung-budget", type=int, default=400_000, help="stop if the frontier rung is not cleared within this many steps")
    t.add_argument("--eval-freq", type=int, default=10_000); t.add_argument("--checkpoint-freq", type=int, default=50_000)
    t.add_argument("--min-win-rate", type=float, default=0.75); t.add_argument("--min-margin", type=float, default=0.25)
    t.add_argument("--consecutive", type=int, default=2); t.add_argument("--torch-threads", type=int, default=1)
    e = sub.add_parser("evaluate"); e.add_argument("--runs", nargs="+", required=True); e.add_argument("--ladder", default=str(LADDER_PATH))
    e.add_argument("--extra", action="append", default=[], metavar="NAME=CHECKPOINT"); e.add_argument("--workers", type=int, default=12)
    e.add_argument("--output", default=str(ROOT / "artifacts/heading-ladder-results.json"))
    args = parser.parse_args()
    {"scan": scan, "make": make, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
