"""Write config/training_curriculum_alignment_v1.json: the frozen curriculum plus misaligned near-pit starts.

The frozen curriculum is not modified. The new `near_pit_misaligned` group uses the push-sweep geometry from
scripts/ordinary_start_diagnostic.py (red on the pit diagonal facing blue, blue 0.18 m behind it) with blue's
heading rotated 20, 45 or 90 degrees off the pit line. Always-Advance wins none of those; a turn-then-advance
controller wins nearly all of them. Offsets of 30 and 60 degrees and the 0.80 m range are left out so the
diagnostic sweep keeps held-out cells.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ordinary_start_diagnostic as diagnostic  # noqa: E402

ROOT = diagnostic.ROOT
OUTPUT = ROOT / "config/training_curriculum_alignment_v1.json"
TRAINING_DISTANCES_M = (0.32, 0.45, 0.60)
TRAINING_ABS_OFFSETS_DEG = (20, 45, 90)
# Ordinary starts keep their frozen share in every stage; the new group displaces near-pit starts only.
WEIGHTS = {"near_pit": {"near_pit_easy": 0.45, "near_pit_offset": 0.25, "near_pit_misaligned": 0.30},
           "widening": {"near_pit_easy": 0.15, "near_pit_offset": 0.25, "near_pit_misaligned": 0.35, "ordinary": 0.25},
           "mixed": {"near_pit_easy": 0.10, "near_pit_offset": 0.15, "near_pit_misaligned": 0.25, "ordinary": 0.50}}


def main():
    from autobot.sim.scenarios import validate_curriculum
    frozen = json.loads((diagnostic.FROZEN / "training_curriculum.json").read_text(encoding="utf-8"))
    misaligned = [{"name": case["name"].replace("sweep_", "train_misaligned_"),
                   "player": [round(value, 6) for value in case["player"]], "opponent": [round(value, 6) for value in case["opponent"]]}
                  for case in diagnostic.sweep_scenarios()
                  if case["distance_m"] in TRAINING_DISTANCES_M and abs(case["heading_offset_deg"]) in TRAINING_ABS_OFFSETS_DEG]
    curriculum = {
        "seed": frozen["seed"],
        "description": "alignment_v1: frozen curriculum plus near-pit starts where blue must turn 20-90 degrees before pushing. "
                       "Ordinary-start share per stage is unchanged.",
        "derived_from": "runs/ppo-curriculum-overnight/training_curriculum.json",
        "stages": [dict(stage, group_weights=WEIGHTS[stage["name"]]) for stage in frozen["stages"]],
        "groups": dict(frozen["groups"], near_pit_misaligned=misaligned)}
    validate_curriculum(curriculum)
    for stage, original in zip(curriculum["stages"], frozen["stages"]):
        assert stage["until_episode"] == original["until_episode"] and abs(sum(stage["group_weights"].values()) - 1) < 1e-9
        assert stage["group_weights"].get("ordinary", 0) == original["group_weights"].get("ordinary", 0)
    OUTPUT.write_text(json.dumps(curriculum, indent=1), encoding="utf-8")
    print(OUTPUT, "misaligned starts:", len(misaligned))


if __name__ == "__main__":
    main()
