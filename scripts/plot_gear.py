"""Plot the gear sweep by family (median + inter-quartile band) and print the robust window width.

    python -m scripts.plot_gear results/data/gear_sweep_all.csv

Robust window = width of the contiguous gear range around 1.0 where the return stays > 50.
The sweep stops at 1.5x, so windows that reach this bound are truncated (lower bounds).
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
s = df["scale"].values
fams = {}
for c in df.columns:
    if c != "scale":
        fams.setdefault(re.sub(r"_s\d+(\.zip)?$", "", c), []).append(c)
colors = {"base300k": "green", "base": "blue", "dr": "orange"}
labels = {"base300k": "base, 300k steps", "base": "base, 500k steps", "dr": "domain randomization (gear, mass)"}

plt.figure(figsize=(7, 4.5))
for fam, cs in fams.items():
    v = df[cs].values
    plt.fill_between(s, np.percentile(v, 25, axis=1), np.percentile(v, 75, axis=1),
                     color=colors[fam], alpha=0.2)
    plt.plot(s, np.median(v, axis=1), color=colors[fam], lw=2.5, label=f"{labels[fam]} (n={len(cs)})")
plt.axvline(1.0, ls="--", color="gray", lw=1)
plt.xlabel("motor gain factor (gear)"); plt.ylabel("episode return")
plt.title("Robustness to motor gain (median, inter-quartile band)")
plt.legend(); plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/gear_sweep_families.png", dpi=150)


def window(y, thr=50):
    i0 = int(np.argmin(abs(s - 1.0)))
    if y[i0] < thr:
        return 0.0
    lo = hi = i0
    while lo > 0 and y[lo - 1] > thr: lo -= 1
    while hi < len(y) - 1 and y[hi + 1] > thr: hi += 1
    return round(float(s[hi] - s[lo]), 2)


for fam, cs in fams.items():
    w = [window(df[c].values) for c in cs]
    print(f"{fam:9s} robust window: mean {np.mean(w):.2f}  std {np.std(w):.2f}   per seed {w}")
