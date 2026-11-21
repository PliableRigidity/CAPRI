"""Do a policy's tactic preferences carry over from one pit to the other? Offline; trains nothing.

    python scripts/pit_symmetry_probe.py NAME=CHECKPOINT [NAME=CHECKPOINT ...]

The arena is symmetric under a half-turn about its centre, which swaps the two pits. A start at one pit and its
half-turn image at the other are the same situation for the robot and need the same tactic. In the sim_debug
observation, though, the six pose channels (own x, y, sin yaw, cos yaw, opponent x, y) are measured about the
arena centre, so the image start has exactly the negated pose channels. A tanh network with small biases is
close to an odd function of its input: what it prefers at one pit it tends to avoid at the other.

For matched start pairs this reports, at the first decision:
  preference_similarity   cosine similarity of the centred tactic logits at the two pits
                          (+1: same preferences at both pits, -1: exactly reversed)
  same_greedy_tactic      share of pairs where the greedy tactic is the same at both pits
  greedy advance / P(advance) per pit on the aligned pairs (offset <= 10 deg), where advancing wins at both pits
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import heading_ladder as experiment  # noqa: E402

ROOT = experiment.ROOT
DISTANCES_M = (0.34, 0.46, 0.58)
OFFSETS_DEG = (0, 10, -10, 24, -24, 45, -45)
SEED = 93017


def probe(name, path):
    import torch
    from autobot.sim.cli import checkpoint_observation_metadata, load_checkpoint
    from autobot.sim.env import AutoBotEnv, previous_action_slice
    from autobot.sim.ladder import GreedyActor, ladder_start
    torch.set_num_threads(1)
    model, encoding = load_checkpoint(str(path), device="cpu"); metadata = checkpoint_observation_metadata(path) or {}
    action_profile = metadata.get("action_profile", "legacy5")
    env = AutoBotEnv(str(experiment.CONFIG), "sim_debug", 0, opponent_model="last_year", opponent_mode="scripted",
                     reward_profile="outcome_only", action_profile=action_profile, previous_action_encoding=encoding)
    actor = GreedyActor(model); pose = slice(previous_action_slice(action_profile).stop, None); tactics = env.tactics
    pairs = []; index = 0
    for distance in DISTANCES_M:
        for offset in OFFSETS_DEG:
            rows = {}
            for pit in ("top_left", "bottom_right"):
                observation, _ = env.reset(seed=SEED + index, options={"scenario": ladder_start(pit, distance, offset, "probe")})
                values = actor.logits(observation); probabilities = np.exp(values - values.max()); probabilities /= probabilities.sum()
                rows[pit] = {"observation": observation, "logits": values, "greedy": tactics[int(np.argmax(values))],
                             "p_advance": float(probabilities[tactics.index("advance")])}
            index += 1
            a = rows["top_left"]["logits"] - rows["top_left"]["logits"].mean(); b = rows["bottom_right"]["logits"] - rows["bottom_right"]["logits"].mean()
            pairs.append({"distance_m": distance, "heading_offset_deg": offset,
                          "pose_channels_are_negated": bool(np.allclose(rows["top_left"]["observation"][pose], -rows["bottom_right"]["observation"][pose], atol=1e-4)),
                          "preference_similarity": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
                          **{f"{pit}_greedy": rows[pit]["greedy"] for pit in rows}, **{f"{pit}_p_advance": rows[pit]["p_advance"] for pit in rows}})
    aligned = [pair for pair in pairs if abs(pair["heading_offset_deg"]) <= 10]
    hidden_bias = [float(np.abs(bias).mean()) for _, bias in actor.hidden]
    return name, {"path": experiment.relative(path), "action_profile": action_profile, "previous_action_encoding": encoding,
                  "pairs": len(pairs), "pose_channels_are_negated_in_every_pair": all(pair["pose_channels_are_negated"] for pair in pairs),
                  "mean_preference_similarity": float(np.mean([pair["preference_similarity"] for pair in pairs])),
                  "same_greedy_tactic": float(np.mean([pair["top_left_greedy"] == pair["bottom_right_greedy"] for pair in pairs])),
                  "aligned_pairs": len(aligned),
                  **{f"aligned_greedy_advance_{pit}": int(sum(pair[f"{pit}_greedy"] == "advance" for pair in aligned)) for pit in ("top_left", "bottom_right")},
                  **{f"aligned_mean_p_advance_{pit}": float(np.mean([pair[f"{pit}_p_advance"] for pair in aligned])) for pit in ("top_left", "bottom_right")},
                  "mean_abs_hidden_bias_per_layer": hidden_bias, "mean_abs_output_bias": float(np.abs(actor.output[1]).mean()),
                  "pair_rows": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", metavar="NAME=CHECKPOINT")
    parser.add_argument("--output", default=str(ROOT / "artifacts/pit-symmetry-probe.json"))
    args = parser.parse_args()
    results = {}
    for item in args.checkpoints:
        name, path = item.split("=", 1); path = (ROOT / path) if not Path(path).is_absolute() else Path(path)
        if not path.is_file(): print("missing", path); continue
        results.update([probe(name, path)])
    output = {"purpose": "Do tactic preferences carry over between the two pits, which are half-turn images of each other?",
              "config": experiment.relative(experiment.CONFIG), "distances_m": DISTANCES_M, "heading_offsets_deg": OFFSETS_DEG,
              "definitions": {"preference_similarity": "cosine similarity of centred first-decision tactic logits at a start and at its half-turn image "
                                                       "(+1 same preferences, -1 reversed)",
                              "aligned": "offsets of 10 deg or less, where a plain advance wins at both pits"},
              "policies": results}
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(args.output)
    for name, row in results.items():
        print(f"{name:44s} similarity {row['mean_preference_similarity']:+.2f}  same greedy {row['same_greedy_tactic']:.2f}  "
              f"aligned greedy advance TL {row['aligned_greedy_advance_top_left']}/{row['aligned_pairs']} BR {row['aligned_greedy_advance_bottom_right']}/{row['aligned_pairs']}  "
              f"P(advance) TL {row['aligned_mean_p_advance_top_left']:.2f} BR {row['aligned_mean_p_advance_bottom_right']:.2f}  negated {row['pose_channels_are_negated_in_every_pair']}")


if __name__ == "__main__":
    main()
