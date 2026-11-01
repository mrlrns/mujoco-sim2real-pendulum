"""Compare latency robustness of all families and print a summary table.

    python -m scripts.plot_compare
Reads results/data/latency_standard.csv and results/data/latency_prev_action.csv.
Thin lines = individual seeds, thick lines = median over seeds.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--standard", default="results/data/latency_standard.csv")
ap.add_argument("--prev", default="results/data/latency_prev_action.csv")
ap.add_argument("--out", default="results/figures/latency_all.png")
a = ap.parse_args()

std, prv = pd.read_csv(a.standard), pd.read_csv(a.prev)
ms = std["ms"].values


def family(df, fam):
    cs = sorted(c for c in df.columns if re.fullmatch(rf"{fam}_s\d+", c))
    return np.array([df[c].values for c in cs]).T if cs else None   # (delays, seeds)


# (column prefix, label, colour, dataframe)
PANELS = [
    ("observation: [cos, sin, omega]", [
        ("base", "base (nominal)", "blue", std), ("dr", "dr (gear, mass)", "orange", std),
        ("lat2", "lat2 (delay 0-2 steps)", "green", std), ("lat5", "lat5 (delay 0-5 steps)", "red", std),
        ("full", "full (gear, mass, delay 0-3)", "purple", std)]),
    ("observation: [cos, sin, omega, previous action]", [
        ("apbase", "base", "blue", prv), ("apdr", "dr (gear, mass)", "orange", prv),
        ("aplat3", "lat3 (delay 0-3 steps)", "green", prv), ("aplat5", "lat5 (delay 0-5 steps)", "red", prv),
        ("apfull", "full (gear, mass, delay 0-3)", "purple", prv)]),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
rows = []
for ax, (title, fams) in zip(axes, PANELS):
    for key, label, c, df in fams:
        v = family(df, key)
        if v is None:
            continue
        for j in range(v.shape[1]):
            ax.plot(ms, v[:, j], color=c, alpha=0.12, lw=1)
        ax.plot(ms, np.median(v, axis=1), color=c, lw=3, label=f"{label} (n={v.shape[1]})")
        med = np.median(v, axis=1)
        thr = next((int(ms[i]) for i in range(len(ms)) if med[i] < 50), ">200")
        i100 = int(np.argmin(abs(ms - 100)))
        rows.append(("standard" if df is std else "+prev action",
                     label, med[0], thr, int((v[0] < 50).sum()), int((v[i100] < 50).sum()), v.shape[1]))
    ax.set_title(title); ax.set_xlabel("action latency (ms)"); ax.legend(loc="lower left")
axes[0].set_ylabel("episode return")
plt.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
plt.savefig(a.out, dpi=150)

print(f"{'obs':<14}{'family':<30}{'nominal':>8}{'thr (ms)':>10}{'failed@0ms':>12}{'<50@100ms':>11}")
for obs, label, nom, thr, f0, f100, n in rows:
    print(f"{obs:<14}{label:<30}{nom:>8.0f}{str(thr):>10}{f0:>9}/{n:<2}{f100:>8}/{n}")
