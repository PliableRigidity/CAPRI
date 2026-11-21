"""Read-only replays of recorded simulator rounds, for the case-study figures.

Reads the play logs under runs/ and replays them through the unchanged simulator with the frozen
settings, using the helper functions in scripts/live_round_diagnostic.py. Nothing is trained and
nothing under runs/, artifacts/, config/ or autobot/ is written.

Outputs (both inside docs/case-study/tools/.cache/, which is not meant to be committed):
  traces.npz             pose traces for the figures
  live-sessions.json     per-round summary of the live play logs

Run from the repository root:  python docs/case-study/tools/replay_traces.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
CACHE = Path(__file__).resolve().parent / ".cache"

import live_round_diagnostic as lrd  # noqa: E402

REFERENCE = "runs/combined-clone-balanced-dagger-seed2/final_model.zip"
CHALLENGER = "runs/recovery-challenger-v2-seed1/final_model.zip"
LOGS = [
    ("reference", REFERENCE, "runs/overnight-2026-11-17/live/reference-normal-spawn-20261118-000023.jsonl"),
    ("challenger", CHALLENGER, "runs/reactive-recovery-2026-11-18/live/challenger-normal-spawn-20261118-095851.jsonl"),
    ("challenger", CHALLENGER, "runs/reactive-recovery-2026-11-18/live/challenger-normal-spawn-20261118-202646.jsonl"),
]
STEP_S = lrd.STEP_S


def summarise(trace, steps):
    """Per-round facts from a replayed trace and the logged steps it was replayed from."""
    tactics = [lrd.TACTICS[a] for a in trace["actions"]]
    logged = [s["blue_tactic"] for s in steps]
    contact = np.asarray(trace["contact"], dtype=bool)
    red = [tuple(s["red_command"]) for s in steps]
    first = int(np.argmax(contact)) if contact.any() else None
    one_way = lrd.longest_one_way_rotation(trace)
    out = {
        "decisions": len(steps),
        "seconds": round(len(steps) * STEP_S, 2),
        "logged_end": steps[-1]["end_reason"],
        "replayed_end": lrd.outcome(trace),
        "blue_tactics_reproduced": f"{sum(a == b for a, b in zip(tactics, logged))}/{len(logged)}",
        "red_idle_share": round(sum(c == (0.0, 0.0) for c in red) / len(red), 3),
        "first_contact_s": None if first is None else round(first * STEP_S, 2),
        "contact_s": round(float(contact.sum()) * STEP_S, 2),
        "contact_in_last_half_second": bool(contact[-25:].any()),
        "longest_one_way_rotation_deg": one_way["rotation_deg"],
        "closest_m": round(float(np.linalg.norm(trace["poses"][:, :2] - trace["poses"][:, 3:5], axis=1).min()), 3),
    }
    return out


def main():
    CACHE.mkdir(exist_ok=True)
    arrays = {}
    report = {"note": "Replays of recorded play logs through the unchanged simulator (seed 10007, frozen 20 s config, normal spawn). "
                      "Red is the logged command stream. 'blue_tactics_reproduced' compares the replayed blue with the logged blue.",
              "sessions": []}
    sessions = {}
    for label, model, log in LOGS:
        if model not in sessions:
            sessions[model] = lrd.make_session(model=model)
        session = sessions[model]
        env = session.env
        blue = lrd.FastClone(session.model)
        stop = lrd.Stop()
        rounds = lrd.load_rounds(ROOT / log)
        entry = {"log": log, "blue": model, "rounds": []}
        for number, steps in enumerate(rounds, 1):
            red = [tuple(s["red_command"]) for s in steps]
            trace = lrd.rollout(env, blue, red)
            item = {"round": number, **summarise(trace, steps)}
            if item["replayed_end"] == "red falls into a pit":
                # attribution control: does red fall with the same commands against a blue that never moves?
                alone = lrd.rollout(env, stop, red)
                item["red_falls_against_a_motionless_blue"] = lrd.outcome(alone) == "red falls into a pit"
            entry["rounds"].append(item)
            key = f"{Path(log).stem}__round{number}"
            arrays[key + "__poses"] = trace["poses"]
            arrays[key + "__actions"] = trace["actions"]
            arrays[key + "__contact"] = np.asarray(trace["contact"], dtype=bool)
        report["sessions"].append(entry)

    # Fixed-input comparison: the recorded spin round (round 2 of the reference session) against both blues.
    spin_steps = lrd.load_rounds(ROOT / LOGS[0][2])[1]
    red = [tuple(s["red_command"]) for s in spin_steps]
    arrays["spin_red_commands"] = np.asarray(red, dtype=float)
    for name, model in (("reference", REFERENCE), ("challenger", CHALLENGER)):
        session = sessions[model]
        trace = lrd.rollout(session.env, lrd.FastClone(session.model), red)
        arrays[f"spin__{name}__poses"] = trace["poses"]
        arrays[f"spin__{name}__actions"] = trace["actions"]
        arrays[f"spin__{name}__contact"] = np.asarray(trace["contact"], dtype=bool)
        report.setdefault("fixed_input_spin_round", {})[name] = {"blue": model, **summarise(trace, spin_steps)}
        if name == "challenger":
            # against a different blue the logged tactics are not expected to match
            report["fixed_input_spin_round"][name].pop("blue_tactics_reproduced")

    np.savez_compressed(CACHE / "traces.npz", **arrays)
    (CACHE / "live-sessions.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
