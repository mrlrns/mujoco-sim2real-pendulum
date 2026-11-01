"""Sweep the motor gain factor (0.5x .. 1.5x) for several models and save the returns to a CSV.

    python -m scripts.sweep_gear --mods models/base_s*.zip models/dr_s*.zip --tag all
"""
import argparse
import csv
import os

import numpy as np
from stable_baselines3 import PPO

from sim2real.evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--mods", nargs="+", required=True, help="one or more model files")
ap.add_argument("--n_ep", type=int, default=10, help="episodes per point")
ap.add_argument("--tag", default="all")
a = ap.parse_args()

scales = np.arange(0.5, 1.51, 0.01)
results = {}
for path in a.mods:
    name = os.path.basename(path).replace(".zip", "")      # e.g. base_s0
    model = PPO.load(path, device="cpu")
    results[name] = [evaluate(model, "gear", s, n_ep=a.n_ep)[0] for s in scales]
    print(name, "done")

os.makedirs("results/data", exist_ok=True)
with open(f"results/data/gear_sweep_{a.tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scale"] + list(results))
    for i, s in enumerate(scales):
        w.writerow([round(float(s), 2)] + [round(results[n][i], 1) for n in results])
