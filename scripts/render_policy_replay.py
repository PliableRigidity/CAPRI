"""Render a short deterministic MuJoCo replay to MP4 for experiment artifacts."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import mujoco
from PIL import Image
from stable_baselines3 import PPO

from autobot.sim.env import ACTION_PROFILES, AutoBotEnv
from autobot.sim.scenarios import load_scenario_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--action-profile", choices=ACTION_PROFILES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20017)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "runs/ppo-curriculum-overnight/effective_sim.yaml"
    scenarios = load_scenario_file(root / "runs/ppo-curriculum-overnight/evaluation_scenarios.json")
    scenario = next(item for item in scenarios if item["name"] == "default_spawn")
    env = AutoBotEnv(str(config), "sim_debug", args.seed, opponent_model="last_year",
                     opponent_mode="scripted", reward_profile="outcome_only", action_profile=args.action_profile)
    obs, _ = env.reset(seed=args.seed, options={"scenario": scenario})
    model = PPO.load(args.model, device="cpu")
    renderer = mujoco.Renderer(env.sim.model, height=720, width=720)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="autobot-replay-") as temporary:
        directory = Path(temporary)
        frame = 0
        while env.sim.data.time < args.seconds:
            action = int(model.predict(obs, deterministic=True)[0])
            obs, _, terminated, truncated, _ = env.step(action)
            if frame % 2 == 0:
                renderer.update_scene(env.sim.data, camera="overhead")
                Image.fromarray(renderer.render()).save(directory / f"frame-{frame // 2:05d}.png")
            frame += 1
            if terminated or truncated:
                break
        renderer.close()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25",
                        "-i", str(directory / "frame-%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(output)], check=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
