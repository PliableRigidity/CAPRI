"""Offline mobility/jitter metrics shared by the analysis scripts. Never used for reward or selection."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .env import AutoBotEnv

# A step is "stationary" when the chassis centre has net-moved less than this over the trailing
# window (under 2 cm/s on average; full speed is about 68 cm/s). Unlike a per-step speed test this
# also catches a bot that jitters in place, and unlike the wall metric it needs no wall contact.
STATIONARY_WINDOW_S = 0.5
STATIONARY_NET_DISPLACEMENT_M = 0.010


def period_two_runs(actions, minimum_steps=4):
    """Maximal A,B,A,B... runs (A != B) of at least two full cycles."""
    runs, start = [], None
    for index in range(2, len(actions) + 1):
        alternating = index < len(actions) and actions[index] == actions[index - 2] and actions[index] != actions[index - 1]
        if alternating and start is None:
            start = index - 2
        if not alternating and start is not None:
            if index - start >= minimum_steps: runs.append((start, index))
            start = None
    return runs


def mobility_metrics(start_xy, positions_xy, actions, contacts, stationary_wall, control_hz, tactics):
    """Metrics for one episode. ``positions_xy[i]`` is the blue centre after decision ``i``."""
    positions = np.asarray(positions_xy, dtype=float).reshape(-1, 2)
    start = np.asarray(start_xy, dtype=float)
    steps = len(actions); seconds = steps / control_hz
    window = round(STATIONARY_WINDOW_S * control_hz)
    trail = np.vstack([start, positions])                       # trail[i] = position before decision i
    reference = trail[np.maximum(np.arange(steps) + 1 - window, 0)]
    stationary = np.linalg.norm(positions - reference, axis=1) < STATIONARY_NET_DISPLACEMENT_M
    contact_index = next((index for index, flag in enumerate(contacts) if flag), None)
    pre = steps if contact_index is None else contact_index + 1
    switches = sum(a != b for a, b in zip(actions, actions[1:]))
    cycle_steps = sum(end - begin for begin, end in period_two_runs(actions))
    return {
        "steps": steps, "simulated_seconds": seconds,
        "tactic_switches": switches, "switches_per_s": switches / max(seconds, 1e-9),
        "period_2_cycle_fraction": cycle_steps / max(steps, 1),
        "time_to_first_contact_s": None if contact_index is None else pre / control_hz,
        "pre_contact_net_displacement_m": float(np.linalg.norm(positions[pre - 1] - start)),
        "pre_contact_path_m": float(np.linalg.norm(np.diff(trail[:pre + 1], axis=0), axis=1).sum()),
        "total_path_m": float(np.linalg.norm(np.diff(trail, axis=0), axis=1).sum()),
        "stationary_time_s": float(stationary.sum() / control_hz),
        "pre_contact_stationary_time_s": float(stationary[:pre].sum() / control_hz),
        "stationary_wall_time_s": float(sum(stationary_wall) / control_hz),
        "contact_time_s": float(sum(contacts) / control_hz),
        "action_counts": dict(Counter(tactics[int(action)] for action in actions)),
    }


def rollout_episode(model, config, scenario, *, profile, action_profile, previous_action_encoding, seed,
                    opponent_mode="scripted", opponent_chassis="last_year", red_commands=None,
                    deterministic=True, keep_trace=False, on_env=None):
    """One reproducible episode. ``red_commands`` replays recorded red wheel commands open-loop.

    ``deterministic=False`` samples tactics the way training does (seeded, so still reproducible);
    ``keep_trace`` adds a per-decision pose trace for inspection.
    """
    if not deterministic:
        import torch
        torch.manual_seed(int(seed))
    env = AutoBotEnv(str(config), profile, seed, opponent_model=opponent_chassis, opponent_mode=opponent_mode,
                     reward_profile="outcome_only", action_profile=action_profile,
                     previous_action_encoding=previous_action_encoding)
    if on_env is not None: on_env(env)   # lets a privileged diagnostic controller read simulator truth
    obs, _ = env.reset(seed=seed, options={"scenario": scenario})
    start = env.sim.truth(0)["position_m"][:2].copy()
    opponent_start = env.sim.truth(1)["position_m"][:2].copy()
    positions, actions, contacts, walls, probabilities = [], [], [], [], []
    arena = env.sim.cfg["arena"]; pits = np.asarray(arena["pit_centres_m"], dtype=float)
    pit_edge = lambda xy: float(np.min(np.linalg.norm(pits - xy, axis=1)) - arena["pit_radius_m"])
    opponent_pit_edge = [pit_edge(opponent_start)]; opponent_positions = [opponent_start.copy()]
    info = {}; terminated = truncated = False; step = 0
    while True:
        if red_commands is not None and step >= len(red_commands): break
        action = int(model.predict(obs, deterministic=deterministic)[0])
        if hasattr(model, "policy"):
            # Read-only: how decisively the greedy tactic beats the alternatives at this state.
            tensor, _ = model.policy.obs_to_tensor(obs)
            probabilities.append(model.policy.get_distribution(tensor).distribution.probs.detach().cpu().numpy()[0])
        override = None if red_commands is None else tuple(red_commands[step])
        obs, _, terminated, truncated, info = env.step(action, opponent_drive_override=override)
        positions.append(env.sim.truth(0)["position_m"][:2].copy()); actions.append(action)
        contacts.append(bool(info["opponent_contact"])); walls.append(bool(info["stationary_wall_contact"]))
        opponent_positions.append(env.sim.truth(1)["position_m"][:2].copy()); opponent_pit_edge.append(pit_edge(opponent_positions[-1]))
        step += 1
        if terminated or truncated: break
    reason = info.get("round_end_reason")
    if reason == "Pit fall": outcome = "own_falls" if info["round_end_bot"] == 0 else "wins"
    elif reason == "Simultaneous pit fall": outcome = "draws"
    elif reason == "Timeout": outcome = "timeouts"
    elif terminated or truncated: outcome = "simulation_bugs"
    else: outcome = "recording_end"
    result = mobility_metrics(start, positions, actions, contacts, walls, env.sim.cfg["simulation"]["control_hz"], env.tactics)
    opponent_displacement = float(np.linalg.norm(env.sim.truth(1)["position_m"][:2] - opponent_start))
    timeout_class = None
    if outcome == "timeouts":
        timeout_class = ("never_reached_red" if result["time_to_first_contact_s"] is None
                         else "contact_stall" if opponent_displacement < 0.05 else "moved_red_no_finish")
    if probabilities:
        ranked = np.sort(np.asarray(probabilities), axis=1)
        margin = ranked[:, -1] - ranked[:, -2]
        result["decision_confidence"] = {
            "mean_greedy_probability": float(ranked[:, -1].mean()),
            "median_top2_probability_margin": float(np.median(margin)),
            "fraction_of_decisions_with_top2_margin_below_0.10": float((margin < 0.10).mean())}
    # How close red's centre came to the nearest pit opening (negative: over the opening).
    result.update(opponent_start_pit_edge_distance_m=opponent_pit_edge[0],
                  opponent_min_pit_edge_distance_m=float(min(opponent_pit_edge)),
                  opponent_final_pit_edge_distance_m=opponent_pit_edge[-1],
                  opponent_path_m=float(np.linalg.norm(np.diff(np.asarray(opponent_positions), axis=0), axis=1).sum()),
                  deterministic=bool(deterministic))
    if keep_trace:
        result["trace"] = {"blue_xy": [[round(float(v), 4) for v in xy] for xy in [start, *positions]],
                           "red_xy": [[round(float(v), 4) for v in xy] for xy in opponent_positions],
                           "tactic": [env.tactics[a] for a in actions], "contact": contacts}
    result.update(scenario=scenario["name"], group=scenario.get("group", "ungrouped"), outcome=outcome,
                  end_reason=reason, opponent_displacement_m=opponent_displacement, timeout_class=timeout_class,
                  previous_action_encoding=env.previous_action_encoding)
    return result
