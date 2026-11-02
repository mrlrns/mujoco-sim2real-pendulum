"""Render a trained policy to a video.

    python -m scripts.watch --model models_demo/dr_s0.zip --out demo.mp4
    python -m scripts.watch --model models_demo/dr_s0.zip --gear 0.9 --delay 4 --out demo_perturbed.mp4
    python -m scripts.watch --model models_demo/full_s0.zip --out assets/demo_nominal.gif   # GIF for the README
A .gif output is subsampled (every 2nd frame, 25 fps) and downscaled to 480 px wide.
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
ap.add_argument("--width", type=int, default=480, help="GIF width in pixels")
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

if a.out.lower().endswith(".gif"):
    from PIL import Image
    h = round(a.width * frames[0].shape[0] / frames[0].shape[1])
    small = [Image.fromarray(f).resize((a.width, h), Image.LANCZOS) for f in frames[::2]]
    small[0].save(a.out, save_all=True, append_images=small[1:], duration=40, loop=0, optimize=True)
else:
    imageio.mimsave(a.out, frames, fps=50)   # 50 control steps = 1 s of simulated time
print(f"episode return: {total:.1f} -> {a.out}")
