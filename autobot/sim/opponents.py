"""Harder scripted red opponents: seeded, parameterised driving styles. Simulation-only.

Nothing here changes the frozen environment. The original opponent modes of `env.py` (`scripted`, `static`,
`faster_follow`, `stop_and_go`, `direction_changes`) are untouched and behave exactly as before. A style is a
separate red driver whose commands are handed to `AutoBotEnv.step` through its existing override argument, so
physics, drive limits (wheel commands in [-1, 1]), observations and the 20 s horizon are those of the frozen
simulator. Blue's observation never contains red's style, its parameters or its future commands.

Styles
    persistent_follower  chases faster than `scripted` and keeps driving at close range (no yield distance)
    curved_approach      steers while moving and comes in on one of blue's flanks, tracking blue's heading
    stop_and_go          varies its pace: drives, pauses, sometimes backs away, then resumes
    reactive_changes     changes course in response to blue's movement after a short variable delay, and
                         commits to each manoeuvre for a while
    pit_pusher           (held out of training and selection) tries to get behind blue and push it into a pit

Every case resolves its parameters from a seed and stores them, so an episode can be replayed exactly.
"""
from __future__ import annotations

import math

import numpy as np

from .env import AutoBotEnv

STYLES = ("persistent_follower", "curved_approach", "stop_and_go", "reactive_changes")
HELDOUT_STYLES = ("pit_pusher",)
ALL_STYLES = STYLES + HELDOUT_STYLES
# Each style has two key parameters split into a low and a high half. Training and validation cases use three of
# the four combinations; the fourth appears only in final test cases.
COMBOS = ("low_low", "high_high", "low_high", "high_low")
SEEN_COMBOS = COMBOS[:3]; UNSEEN_COMBO = COMBOS[3]
CLOSE_RANGE_M = 0.22            # centre distance below which the two chassis are touching or about to
SPECS = {
    "persistent_follower": {"keys": ("speed", "align_tolerance_deg"),
                            "ranges": {"speed": (0.55, 1.0), "align_tolerance_deg": (8.0, 25.0), "turn_command": (0.4, 0.8),
                                       "creep": (0.0, 0.35), "pit_margin_m": (-0.02, 0.05)}},
    "curved_approach": {"keys": ("speed", "flank_offset_m"),
                        "ranges": {"speed": (0.55, 1.0), "flank_offset_m": (0.06, 0.18), "steer_gain": (1.2, 3.0),
                                   "turn_limit": (0.5, 0.9), "pit_margin_m": (-0.02, 0.05)}, "side": True},
    "stop_and_go": {"keys": ("speed", "move_hold_s"),
                    "ranges": {"speed": (0.5, 1.0), "move_hold_s": (0.35, 1.0), "pause_probability": (0.25, 0.5),
                               "back_probability": (0.15, 0.35), "reverse_speed": (0.35, 0.65), "pit_margin_m": (-0.02, 0.05)}},
    "reactive_changes": {"keys": ("reaction_delay_s", "move_hold_s"),
                         "ranges": {"reaction_delay_s": (0.12, 0.45), "move_hold_s": (0.4, 1.2), "speed": (0.55, 1.0),
                                    "yaw_trigger_deg": (15.0, 35.0), "pit_margin_m": (-0.02, 0.05)}},
    "pit_pusher": {"keys": ("speed", "standoff_m"),
                   "ranges": {"speed": (0.6, 1.0), "standoff_m": (0.16, 0.24), "steer_gain": (1.5, 3.0),
                              "turn_limit": (0.6, 0.9), "pit_margin_m": (0.0, 0.05)}},
}


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def mix(forward, turn):
    """(left, right) wheel commands for forward speed and turn rate in [-1, 1]; positive turn is anticlockwise.

    Negative wheel command is chassis-forward in this model (as in `env.tactic_to_drive`). The pair is scaled
    down together when it would exceed the drive limit, so a curve keeps its shape.
    """
    left = -forward + turn; right = -forward - turn; scale = max(1.0, abs(left), abs(right))
    return float(left / scale), float(right / scale)


def unmix(command):
    """(forward, turn) of a (left, right) wheel command."""
    return -(command[0] + command[1]) / 2, (command[0] - command[1]) / 2


def resolve_parameters(style, combo, rng):
    """Draw one parameter set. The two key parameters come from the half-ranges named by `combo`."""
    if style not in SPECS: raise ValueError(f"Unknown opponent style: {style}")
    if combo not in COMBOS: raise ValueError(f"Unknown parameter combination: {combo}")
    spec = SPECS[style]; halves = dict(zip(spec["keys"], combo.split("_"))); parameters = {}
    for name, (low, high) in spec["ranges"].items():
        middle = (low + high) / 2
        if halves.get(name) == "low": high = middle
        elif halves.get(name) == "high": low = middle
        parameters[name] = round(float(rng.uniform(low, high)), 4)
    if spec.get("side"): parameters["side"] = int(rng.choice([-1, 1]))
    return parameters


def combo_of(style, parameters):
    """Which combination a parameter set belongs to (used to check what a split contains)."""
    spec = SPECS[style]
    return "_".join("low" if parameters[name] < sum(spec["ranges"][name]) / 2 else "high" for name in spec["keys"])


def sample_opponent(pool_seed, episode, styles=STYLES, combos=SEEN_COMBOS):
    """Seeded draw of a style, its parameters and its driver seed for one episode (depends only on the two numbers)."""
    rng = np.random.default_rng([int(pool_seed), int(episode)])
    style = str(rng.choice(list(styles))); combo = str(rng.choice(list(combos)))
    return {"style": style, "combo": combo, "parameters": resolve_parameters(style, combo, rng), "driver_seed": int(rng.integers(2 ** 31))}


class RedDriver:
    """One red driving style with fixed parameters. `drive(sim)` returns the wheel commands for this control step."""

    def __init__(self, style, parameters, seed):
        if style not in SPECS: raise ValueError(f"Unknown opponent style: {style}")
        self.style = style; self.p = dict(parameters); self.seed = int(seed); self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed); self.state = {}

    # ----- shared pieces

    @staticmethod
    def _error(red, target):
        position = red["position_m"][:2]
        return wrap(math.atan2(target[1] - position[1], target[0] - position[0]) - red["yaw_rad"])

    def _seek(self, red, target, speed, tolerance_deg, turn_command, creep=0.0):
        """Turn towards the target, then drive at it. `creep` keeps some forward motion during moderate turns."""
        error = self._error(red, target)
        if abs(error) <= math.radians(tolerance_deg): return speed, 0.0
        forward = creep * speed if abs(error) < math.pi / 2 else 0.0
        return forward, math.copysign(turn_command, error)

    def _curve(self, red, target, speed, gain, limit):
        """Steer while moving: forward speed falls with heading error but never reaches zero."""
        error = self._error(red, target)
        return speed * max(0.3, math.cos(error)), float(np.clip(gain * error, -1.0, 1.0)) * limit

    def _guard(self, sim, red, forward, turn):
        """Do not drive into a pit: if the next 9 cm of travel ends over one, turn towards the arena centre instead."""
        if abs(forward) < 1e-6: return forward, turn
        arena = sim.cfg["arena"]; yaw = red["yaw_rad"]; direction = math.copysign(1.0, forward)
        ahead = red["position_m"][:2] + direction * 0.09 * np.array([math.cos(yaw), math.sin(yaw)])
        edge = min(math.dist(ahead, pit) for pit in arena["pit_centres_m"]) - arena["pit_radius_m"]
        if edge >= self.p.get("pit_margin_m", 0.0): return forward, turn
        centre = (arena["width_m"] / 2, arena["height_m"] / 2)
        return 0.0, math.copysign(0.6, self._error(red, centre) or 1.0)

    # ----- styles

    def _persistent_follower(self, sim, red, blue, now):
        return self._seek(red, blue["position_m"][:2], self.p["speed"], self.p["align_tolerance_deg"], self.p["turn_command"], self.p["creep"])

    def _flank_point(self, red, blue, offset, side):
        position = blue["position_m"][:2]
        if math.dist(position, red["position_m"][:2]) < 0.13: return position
        yaw = blue["yaw_rad"]
        return position + side * offset * np.array([-math.sin(yaw), math.cos(yaw)])       # a point off blue's own left or right side

    def _curved_approach(self, sim, red, blue, now):
        target = self._flank_point(red, blue, self.p["flank_offset_m"], self.p["side"])
        return self._curve(red, target, self.p["speed"], self.p["steer_gain"], self.p["turn_limit"])

    def _stop_and_go(self, sim, red, blue, now):
        state = self.state
        if not state or now >= state["until"]:
            hold = self.p["move_hold_s"]; previous = state.get("kind")
            draw = self.rng.random()
            if previous == "go" and draw < self.p["pause_probability"]: kind, length = "pause", self.rng.uniform(0.5, 1.5) * hold
            elif previous == "go" and draw < self.p["pause_probability"] + self.p["back_probability"]: kind, length = "back", self.rng.uniform(0.6, 1.2) * hold
            else: kind, length = "go", self.rng.uniform(1.0, 2.5) * hold
            state.update(kind=kind, until=now + length, pace=float(self.rng.uniform(0.6, 1.0)), veer=float(self.rng.uniform(-0.3, 0.3)))
        if state["kind"] == "pause": return 0.0, 0.0
        if state["kind"] == "back": return -self.p["reverse_speed"], state["veer"]
        return self._seek(red, blue["position_m"][:2], self.p["speed"] * state["pace"], 12.0, 0.6, 0.2)

    def _reactive_changes(self, sim, red, blue, now):
        state = self.state; position = blue["position_m"][:2]
        if not state: state.update(mode="follow", commit_until=0.0, pending=None, yaw=blue["yaw_rad"], position=position.copy(), side=1)
        moved = (abs(wrap(blue["yaw_rad"] - state["yaw"])) > math.radians(self.p["yaw_trigger_deg"])
                 or math.dist(position, state["position"]) > 0.10)
        if moved and state["pending"] is None and now >= state["commit_until"]:
            # Blue did something: react, but only after a short delay that varies from one reaction to the next.
            state.update(pending=now + self.p["reaction_delay_s"] * self.rng.uniform(0.7, 1.3), yaw=blue["yaw_rad"], position=position.copy())
        if state["pending"] is not None and now >= state["pending"]:
            facing = abs(wrap(math.atan2(red["position_m"][1] - position[1], red["position_m"][0] - position[0]) - blue["yaw_rad"]))
            spin = float(blue["angular_velocity_rad_s"][2])
            side = -int(math.copysign(1, spin)) if abs(spin) > 0.5 else int(self.rng.choice([-1, 1]))
            if facing < math.radians(35): mode = "retreat" if self.rng.random() < 0.35 else "sidestep"      # blue is lining red up
            elif facing > math.radians(100): mode = "charge"                                                # blue has turned away
            else: mode = "cut_in"
            state.update(mode=mode, side=side, pending=None, commit_until=now + self.p["move_hold_s"] * self.rng.uniform(0.8, 1.3))
        elif state["mode"] != "follow" and now >= state["commit_until"] and state["pending"] is None:
            state["mode"] = "follow"
        speed = self.p["speed"]; mode = state["mode"]
        if mode == "sidestep": return speed, 0.55 * state["side"]
        if mode == "retreat": return -0.6, 0.3 * state["side"]
        if mode == "charge": return self._seek(red, position, min(1.0, 1.25 * speed), 20.0, 0.7, 0.3)
        if mode == "cut_in": return self._curve(red, self._flank_point(red, blue, 0.14, state["side"]), speed, 2.0, 0.8)
        return self._seek(red, position, speed, 14.0, 0.6, 0.2)

    def _pit_pusher(self, sim, red, blue, now):
        arena = sim.cfg["arena"]; own = red["position_m"][:2]; target = blue["position_m"][:2]
        pit = np.asarray(min(arena["pit_centres_m"], key=lambda centre: math.dist(target, centre)), dtype=float)
        push = (pit - target) / max(float(np.linalg.norm(pit - target)), 1e-9)
        behind = np.clip(target - push * self.p["standoff_m"], 0.09, [arena["width_m"] - 0.09, arena["height_m"] - 0.09])
        to_blue = target - own; distance = max(float(np.linalg.norm(to_blue)), 1e-9)
        if math.dist(behind, own) < 0.07 or (float(np.dot(to_blue / distance, push)) > math.cos(math.radians(18)) and distance < 0.32):
            goal = target + push                                   # lined up behind blue: drive through it towards the pit
        else:
            goal = behind; path = goal - own; length = max(float(np.linalg.norm(path)), 1e-9); along = float(np.dot(to_blue, path / length))
            closest = own + path / length * min(max(along, 0.0), length)
            if 0.0 < along < length and math.dist(target, closest) < 0.19:      # blue is in the way: go round it
                normal = np.array([-push[1], push[0]]); side = normal if float(np.dot(own - target, normal)) >= 0 else -normal
                goal = target + side * 0.27
        return self._curve(red, goal, self.p["speed"], self.p["steer_gain"], self.p["turn_limit"])

    def drive(self, sim):
        red = sim.truth(1); blue = sim.truth(0)
        forward, turn = getattr(self, "_" + self.style)(sim, red, blue, float(sim.data.time))
        return mix(*self._guard(sim, red, forward, turn))


class StyledOpponentEnv(AutoBotEnv):
    """AutoBotEnv whose red can be driven by a `RedDriver`.

    Without an `opponent` reset option and without a sampler it is exactly AutoBotEnv. A reset option
    `opponent={"style", "parameters", "driver_seed"}` replays one resolved opponent. With
    `opponent_sampler={"seed", "styles", "combos"}` a style is sampled at every reset that names no opponent,
    indexed by episode, and the resolved draw is returned in the reset info as `resolved_opponent` so it can
    be saved and replayed. The observation is unchanged in every case.
    """

    def __init__(self, *args, opponent_sampler=None, **kwargs):
        super().__init__(*args, **kwargs); self.red_driver = None; self.resolved_opponent = None
        self.opponent_sampler = dict(opponent_sampler) if opponent_sampler else None; self.opponent_episode = 0

    def reset(self, *, seed=None, options=None):
        options = dict(options or {}); explicit = "opponent" in options; opponent = options.pop("opponent", None)
        observation, info = super().reset(seed=seed, options=options)
        if not explicit and self.opponent_sampler is not None:
            sampler = self.opponent_sampler
            opponent = sample_opponent(sampler["seed"], self.opponent_episode, sampler.get("styles", STYLES), sampler.get("combos", SEEN_COMBOS))
            self.opponent_episode += 1
        self.red_driver = None; self.resolved_opponent = None
        if opponent is not None and opponent.get("style") not in (None, "scripted"):
            self.red_driver = RedDriver(opponent["style"], opponent["parameters"], opponent["driver_seed"])
            self.resolved_opponent = {key: opponent[key] for key in ("style", "combo", "parameters", "driver_seed") if key in opponent}
        info["resolved_opponent"] = self.resolved_opponent
        return observation, info

    def step(self, action, opponent_drive_override=None):
        if opponent_drive_override is None and self.red_driver is not None:
            opponent_drive_override = self.red_driver.drive(self.sim)
        return super().step(action, opponent_drive_override)


def end_outcome(reason, bot, touched_recently):
    """How an episode is counted. Red falling with no recent contact is red's own doing, not blue's win."""
    if reason == "Pit fall" and bot == 1: return "wins" if touched_recently else "red_self_falls"
    if reason == "Pit fall": return "own_falls"
    return {"Simultaneous pit fall": "draws", "Timeout": "timeouts"}.get(reason, "simulation_bugs")


def case_rollout(policy, env, case, keep_commands=False):
    """One reproducible episode of a pool case. Records how it ended and what red actually commanded.

    A red fall counts as blue's win only if the two robots touched within the last half second before it;
    otherwise it is a red self-fall. (The benchmark additionally discounts wins on cases where red also falls
    against a blue that never moves.)
    """
    opponent = None if case["style"] == "scripted" else case
    observation, _ = env.reset(seed=int(case["seed"]), options={"scenario": case["start"], "opponent": opponent})
    if hasattr(policy, "bind"): policy.bind(env)
    sim = env.sim; hz = sim.cfg["simulation"]["control_hz"]; recent = round(0.5 * hz)
    commands = []; distances = []; tactics = []; first_contact = None; last_contact = None; info = {}
    while True:
        action = int(policy.predict(observation, deterministic=True)[0])
        observation, _, terminated, truncated, info = env.step(action)
        commands.append(tuple(float(value) for value in info["opponent_drive"])); distances.append(float(info["distance_to_opponent_m"]))
        tactics.append(env.tactics[action])
        if info["opponent_contact"]:
            last_contact = len(commands)
            if first_contact is None: first_contact = len(commands)
        if terminated or truncated: break
    steps = len(commands); touched = last_contact is not None and steps - last_contact <= recent
    outcome = end_outcome(info.get("round_end_reason"), info.get("round_end_bot"), touched)
    result = {"case": case["name"], "style": case["style"], "family": case.get("family"), "split": case.get("split"), "combo": case.get("combo"),
              "seed": int(case["seed"]), "outcome": outcome, "steps": steps, "simulated_seconds": steps / hz,
              "first_contact_step": first_contact, "contact_within_half_second_of_end": bool(touched),
              "own_fall_while_in_contact": bool(outcome == "own_falls" and touched),
              "blue_tactic_counts": {name: tactics.count(name) for name in env.tactics if name in tactics},
              "red_driving": driving_pattern(commands, distances, hz)}
    if keep_commands: result["red_commands"] = [list(command) for command in commands]; result["blue_tactics"] = tactics; result["distances_m"] = distances
    return result


def driving_pattern(commands, distances, hz):
    """What red's applied wheel commands looked like, overall and at close range."""
    forward = np.array([unmix(command)[0] for command in commands]); turn = np.array([unmix(command)[1] for command in commands])
    distance = np.asarray(distances); close = distance < CLOSE_RANGE_M; steps = len(commands)
    changed = np.array([False] + [max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 0.05 for a, b in zip(commands[1:], commands[:-1])])
    stopped = (np.abs(forward) < 1e-6) & (np.abs(turn) < 1e-6)
    share = lambda mask, within=None: None if (steps if within is None else int(within.sum())) == 0 else float((mask if within is None else mask[within]).mean())
    return {"steps": steps, "close_range_steps": int(close.sum()),
            "close_range_stopped_share": share(stopped, close), "close_range_mean_forward_command": None if not close.any() else float(forward[close].mean()),
            "close_range_driving_forward_share": share(forward > 0.1, close),
            "mean_forward_command_when_driving_forward": None if not (forward > 0.1).any() else float(forward[forward > 0.1].mean()),
            "steering_while_moving_share": share((np.abs(forward) > 0.1) & (np.abs(turn) > 0.1)),
            "reversing_share": share(forward < -0.1), "stopped_share": share(stopped),
            "turning_on_the_spot_share": share((np.abs(forward) <= 0.1) & (np.abs(turn) > 0.1)),
            "command_changes_per_second": float(changed.sum() / max(steps / hz, 1e-9)),
            "mean_seconds_between_command_changes": float(steps / hz / max(int(changed.sum()), 1))}
