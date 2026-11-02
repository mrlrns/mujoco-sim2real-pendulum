"""Out-of-distribution sweeps: perturbations beyond, or outside, the training ranges.

    python -m scripts.sweep_ood --families base,dr,lat5,full --out results/data/ood_standard.csv

One perturbation at a time, standard-observation models:
    mass       x0.5 .. x1.5   (dr and full were trained on x0.8 .. x1.2)
    gear_high  x1.0 .. x2.0   (dr and full were trained on x0.7 .. x1.3)
    obs_noise  sigma 0 .. 0.2 (never randomized for any family; noise std = sigma * [1, 1, 10]
                               on [cos, sin, omega])
Output: long-format CSV with columns test, value, model, family, return.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from sim2real.evaluate import evaluate

TESTS = {
    "mass": [("mass", v, 0.0) for v in np.round(np.arange(0.5, 1.51, 0.1), 2)],
    "gear_high": [("gear", v, 0.0) for v in np.round(np.arange(1.0, 2.01, 0.1), 2)],
    "obs_noise": [("gear", 1.0, s) for s in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]],
}

ap = argparse.ArgumentParser()
ap.add_argument("--families", default="base,dr,lat5,full")
ap.add_argument("--n_ep", type=int, default=10)
ap.add_argument("--out", default="results/data/ood_standard.csv")
a = ap.parse_args()

rows = []
for fam in a.families.split(","):
    for f in sorted(glob.glob(f"models/{fam}_s*.zip")):
        model = PPO.load(f, device="cpu")
        name = os.path.basename(f).replace(".zip", "")
        for test, points in TESTS.items():
            for param, scale, noise in points:
                value = noise if test == "obs_noise" else scale
                ret = evaluate(model, param, scale, n_ep=a.n_ep, obs_noise=noise)[0]
                rows.append((test, float(value), name, fam, round(ret, 1)))
        print(name, "done")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pd.DataFrame(rows, columns=["test", "value", "model", "family", "return"]).to_csv(a.out, index=False)
