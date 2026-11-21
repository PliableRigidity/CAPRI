"""Offline diagnostic helpers: sweep geometry, rollout extras, and the proposed alignment curriculum file."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ordinary_start_diagnostic as diagnostic  # noqa: E402

from autobot.sim.diagnostics import rollout_episode
from autobot.sim.env import ScriptedBaseline, StopBaseline
from autobot.sim.scenarios import validate_curriculum

COMMON = dict(profile="sim_debug", action_profile="legacy5", previous_action_encoding="onehot_v1")
ALIGNMENT = Path("config/training_curriculum_alignment_v1.json")
FROZEN_CURRICULUM = Path("config/training_curriculum.json")


def test_push_sweep_places_red_between_blue_and_the_pit_with_the_stated_heading_offset():
    cases = diagnostic.sweep_scenarios()
    assert len(cases) == 2 * len(diagnostic.SWEEP_DISTANCES_M) * (1 + 2 * (len(diagnostic.SWEEP_HEADING_OFFSETS_DEG) - 1))
    for case in cases:
        pit = np.asarray(diagnostic.PITS[case["pit"]][0]); blue = np.asarray(case["player"][:2]); red = np.asarray(case["opponent"][:2])
        assert np.linalg.norm(red - pit) == pytest.approx(case["distance_m"])
        assert np.linalg.norm(blue - red) == pytest.approx(diagnostic.STANDOFF_M)
        line = math.atan2(pit[1] - red[1], pit[0] - red[0])
        assert math.atan2(red[1] - blue[1], red[0] - blue[0]) == pytest.approx(line)          # red sits on blue's line to the pit
        offset = math.degrees(math.atan2(math.sin(case["player"][2] - line), math.cos(case["player"][2] - line)))
        assert offset == pytest.approx(case["heading_offset_deg"])
        assert 0.05 < blue[0] < 1.10 and 0.05 < blue[1] < 1.10


def test_aligned_push_wins_and_misaligned_push_does_not_for_always_advance():
    config = "config/sim.yaml"
    cases = {(c["pit"], c["distance_m"], c["heading_offset_deg"]): c for c in diagnostic.sweep_scenarios()}
    aligned = rollout_episode(ScriptedBaseline(), config, cases[("top_left", 0.6, 0)], seed=3, **COMMON)
    assert aligned["outcome"] == "wins" and aligned["simulated_seconds"] < 3
    assert aligned["opponent_min_pit_edge_distance_m"] < 0 < aligned["opponent_start_pit_edge_distance_m"] == pytest.approx(0.35)
    env_seen = []
    crooked = rollout_episode(ScriptedBaseline(), config, cases[("top_left", 0.6, 45)], seed=3, keep_trace=True,
                              on_env=env_seen.append, **COMMON)
    # config/sim.yaml uses a 120 s horizon; the outcome that matters here is "no opponent fall".
    assert crooked["outcome"] != "wins" and len(env_seen) == 1
    assert len(crooked["trace"]["blue_xy"]) == crooked["steps"] + 1 == len(crooked["trace"]["red_xy"])
    assert crooked["deterministic"] is True
    stopped = rollout_episode(StopBaseline(), config, cases[("top_left", 0.6, 0)], seed=3, **COMMON)
    assert stopped["outcome"] != "wins"


def test_alignment_curriculum_adds_only_misaligned_near_pit_starts():
    if not ALIGNMENT.is_file(): pytest.skip("alignment curriculum has not been generated")
    proposed = json.loads(ALIGNMENT.read_text(encoding="utf-8")); frozen = json.loads(FROZEN_CURRICULUM.read_text(encoding="utf-8"))
    validate_curriculum(proposed)
    assert {name: proposed["groups"][name] for name in frozen["groups"]} == frozen["groups"]
    assert set(proposed["groups"]) - set(frozen["groups"]) == {"near_pit_misaligned"}
    for new, old in zip(proposed["stages"], frozen["stages"]):
        assert (new["name"], new["until_episode"]) == (old["name"], old["until_episode"])
        assert new["group_weights"].get("ordinary", 0) == old["group_weights"].get("ordinary", 0)
        assert sum(new["group_weights"].values()) == pytest.approx(1.0)
    sweep = {c["name"].replace("sweep_", "train_misaligned_"): c for c in diagnostic.sweep_scenarios()}
    for scenario in proposed["groups"]["near_pit_misaligned"]:
        source = sweep[scenario["name"]]
        assert abs(source["heading_offset_deg"]) in (20, 45, 90) and source["distance_m"] <= 0.60
        assert scenario["player"] == pytest.approx(source["player"], abs=1e-6)
