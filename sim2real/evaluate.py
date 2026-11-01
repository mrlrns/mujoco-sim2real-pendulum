"""Evaluate a trained policy under a fixed perturbation of the simulator."""
import sys

import numpy as np
from stable_baselines3 import PPO

from .env import SwingUpEnv


def evaluate(model, param, scale, n_ep=20):
    """Mean and std of the episode return when ``param`` is set to ``scale``.

    ``param`` is one of "gear", "mass", "damping" (factor of the nominal value)
    or "delay" (absolute number of control steps; 1 step = 20 ms). Only one
    parameter is perturbed at a time. All models see the same start states
    (reset seeds 1000..1000+n_ep-1), so comparisons are paired. The environment
    matches the observation size of the model (with or without previous action).
    """
    env = SwingUpEnv(prev_action=(model.observation_space.shape[0] == 4))  # no randomize: physics fixed below
    env.set_params(**{param: scale})
    returns = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=1000 + ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


if __name__ == "__main__":
    # python -m sim2real.evaluate models/base_s0.zip
    model = PPO.load(sys.argv[1], device="cpu")
    grid = {p: [0.6, 0.8, 1.25, 1.6] for p in ["gear", "mass", "damping"]}

    nominal, _ = evaluate(model, "gear", 1.0)
    print(f"nominal return: {nominal:.1f}")
    for param, scales in grid.items():
        drops = []
        for s in scales:
            mean, _ = evaluate(model, param, s)
            drops.append(nominal - mean)
            print(f"{param:8s} x{s:<5}: drop {nominal - mean:7.1f}")
        print(f"{param:8s} mean drop {np.mean(drops):7.1f} (worst case {max(drops):.1f})\n")
