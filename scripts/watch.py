"""Render a trained policy to a video.

    python -m scripts.watch --model models_demo/dr_s0.zip --out demo.mp4
    python -m scripts.watch --model models_demo/dr_s0.zip --gear 0.9 --delay 4 --out demo_perturbed.mp4
"""
import argparse

import imageio
import mujoco
from stable_baselines3 import PPO

from sim2real.env import SwingUpEnv

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gear", type=float, default=1.0, help="motor gain factor")
ap.add_argument("--mass", type=float, default=1.0)
ap.add_argument("--delay", type=int, default=0, help="action latency in control steps (20 ms each)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="demo.mp4")
a = ap.parse_args()

model = PPO.load(a.model, device="cpu")
env = SwingUpEnv(prev_action=(model.observation_space.shape[0] == 4))
env.set_params(gear=a.gear, mass=a.mass, delay=a.delay)
renderer = mujoco.Renderer(env.model, 720, 1280)

obs, _ = env.reset(seed=a.seed)
frames, total, done = [], 0.0, False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(action)
    total += r
    done = term or trunc
    renderer.update_scene(env.data, camera="cam")
    frames.append(renderer.render())

imageio.mimsave(a.out, frames, fps=50)       # 50 control steps = 1 s of simulated time
print(f"episode return: {total:.1f} -> {a.out}")
