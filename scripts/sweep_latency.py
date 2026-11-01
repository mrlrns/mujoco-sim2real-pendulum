"""Sweep the action latency (0..10 control steps = 0..200 ms) for several model families.

    # standard observation
    python -m scripts.sweep_latency --families base,dr,lat2,lat5,full --out results/data/latency_standard.csv
    # observation with the previous action (models named apbase_s*, apdr_s*, ...)
    python -m scripts.sweep_latency --prefix ap --families base,dr,lat3,lat5,full --out results/data/latency_prev_action.csv
"""
import argparse
import glob
import os

import pandas as pd
from stable_baselines3 import PPO

from sim2real.env import SwingUpEnv
from sim2real.evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--families", default="base,dr,lat2,lat5,full")
ap.add_argument("--prefix", default="", help="model file prefix, e.g. 'ap'")
ap.add_argument("--dmax", type=int, default=10, help="max delay in control steps")
ap.add_argument("--n_ep", type=int, default=10)
ap.add_argument("--out", required=True)
a = ap.parse_args()

e = SwingUpEnv()
dt_ms = e.frame_skip * e.model.opt.timestep * 1000     # one control step = 20 ms
delays = list(range(a.dmax + 1))

data = {"delay": delays, "ms": [d * dt_ms for d in delays]}
for fam in a.families.split(","):
    for f in sorted(glob.glob(f"models/{a.prefix}{fam}_s*.zip")):
        model = PPO.load(f, device="cpu")
        name = os.path.basename(f).replace(".zip", "")
        data[name] = [evaluate(model, "delay", d, n_ep=a.n_ep)[0] for d in delays]
        print(name, "done")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pd.DataFrame(data).to_csv(a.out, index=False)
