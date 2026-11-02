"""Plot the out-of-distribution sweeps and print the range where the median return stays > 50.

    python -m scripts.plot_ood
Reads results/data/ood_standard.csv (made by scripts.sweep_ood). Thin lines = seeds, thick = median.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="results/data/ood_standard.csv")
ap.add_argument("--out", default="results/figures/ood.png")
a = ap.parse_args()

df = pd.read_csv(a.data)
COLORS = {"base": "blue", "dr": "orange", "lat2": "green", "lat5": "red", "full": "purple"}
LABELS = {"base": "base", "dr": "dr (gear, mass)", "lat2": "lat2", "lat5": "lat5 (delay 0-5)",
          "full": "full (gear, mass, delay 0-3)"}
TESTS = [("mass", "mass factor", "trained on 0.8-1.2 (dr, full)", (0.8, 1.2), 1.0),
         ("gear_high", "motor gain factor", "trained on 0.7-1.3 (dr, full)", (1.0, 1.3), 1.0),
         ("obs_noise", "observation noise sigma", "never randomized", None, 0.0)]


def success_range(x, med, x0, thr=50):
    """Contiguous range around x0 where the median return stays above thr."""
    i0 = int(np.argmin(abs(x - x0)))
    if med[i0] < thr:
        return None
    lo = hi = i0
    while lo > 0 and med[lo - 1] > thr: lo -= 1
    while hi < len(x) - 1 and med[hi + 1] > thr: hi += 1
    return float(x[lo]), float(x[hi])


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
print("range where the median return stays > 50 (sweep limits in brackets)")
for ax, (test, xlabel, note, train, x0) in zip(axes, TESTS):
    sub = df[df["test"] == test]
    x_all = np.sort(sub["value"].unique())
    if train:
        ax.axvspan(*train, color="gray", alpha=0.12)
    for fam in [f for f in COLORS if f in set(sub["family"])]:
        piv = sub[sub["family"] == fam].pivot(index="value", columns="model", values="return").sort_index()
        for c in piv.columns:
            ax.plot(piv.index, piv[c], color=COLORS[fam], alpha=0.15, lw=1)
        med = piv.median(axis=1).values
        ax.plot(piv.index, med, color=COLORS[fam], lw=3, label=f"{LABELS[fam]} (n={piv.shape[1]})")
        r = success_range(piv.index.values, med, x0)
        print(f"  {test:10s} {fam:5s} {'none' if r is None else f'{r[0]:.2f} .. {r[1]:.2f}'}"
              f"   [{x_all[0]:.2f} .. {x_all[-1]:.2f}]")
    ax.set_xlabel(xlabel); ax.set_title(note)
axes[0].set_ylabel("episode return"); axes[2].legend(loc="upper right", fontsize=8)
fig.text(0.5, 0.995, "Perturbations outside the training distribution (shaded = training range)", ha="center", va="top")
plt.tight_layout(rect=(0, 0, 1, 0.97))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
plt.savefig(a.out, dpi=150)
print("written:", a.out)
