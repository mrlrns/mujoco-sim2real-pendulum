"""Check that the refactored code reproduces the numbers stored in results/data/*.csv.

    python -m scripts.verify_repro                 # re-evaluates the 3 demo models (~1-2 min)
    python -m scripts.verify_repro --retrain       # also retrains dr seed 0 (500k steps, ~30 s) and compares

Evaluation is deterministic (fixed reset seeds, deterministic policy), so re-evaluating a saved model
should give the CSV values to rounding precision. A retrained model is NOT expected to be bit-identical
(different library versions / hardware), so it is only checked statistically.
"""
import argparse
import os
import subprocess
import sys

import pandas as pd
from stable_baselines3 import PPO

from sim2real.evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--retrain", action="store_true")
ap.add_argument("--tol", type=float, default=0.5, help="max |difference| in return accepted")
a = ap.parse_args()

lat = pd.read_csv("results/data/latency_standard.csv").set_index("delay")
gear = pd.read_csv("results/data/gear_sweep_all.csv").set_index("scale")
gear.index = gear.index.round(2)
gear.columns = [c.replace(".zip", "") for c in gear.columns]   # older CSVs name columns "<model>.zip"
GEAR_POINTS = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]

worst, n = 0.0, 0
for name in ["base_s0", "dr_s0", "full_s0"]:
    m = PPO.load(f"models_demo/{name}.zip", device="cpu")
    for d in range(0, 11):
        new = evaluate(m, "delay", d, n_ep=10)[0]
        old = lat.loc[d, name]
        worst, n = max(worst, abs(new - old)), n + 1
        if abs(new - old) > a.tol:
            print(f"DIFF latency {name} delay={d}: csv {old:.1f} vs now {new:.1f}")
    for s in (GEAR_POINTS if name in gear.columns else []):   # gear sweep CSV has no 'full' family
        new = evaluate(m, "gear", s, n_ep=10)[0]
        old = gear.loc[s, name]
        worst, n = max(worst, abs(new - old)), n + 1
        if abs(new - old) > a.tol:
            print(f"DIFF gear {name} x{s}: csv {old:.1f} vs now {new:.1f}")
    print(name, "checked")
print(f"\n{n} points compared, largest |difference| = {worst:.3f} (tolerance {a.tol})")
ok = worst <= a.tol
print("REPRODUCED" if ok else "NOT REPRODUCED: do not trust the README numbers until this is understood")

if a.retrain:
    out = "/tmp/verify_dr_s0"
    subprocess.run([sys.executable, "-m", "scripts.train", "--family", "dr", "--seed", "0", "--out", out], check=True)
    new = PPO.load(out + ".zip", device="cpu")
    ref = PPO.load("models_demo/dr_s0.zip", device="cpu")
    print("\nretrained dr_s0 vs saved dr_s0 (not expected to be identical):")
    for d in [0, 4, 8, 10]:
        print(f"  delay {d:2d} ({20*d} ms): saved {evaluate(ref, 'delay', d)[0]:7.1f}   retrained {evaluate(new, 'delay', d)[0]:7.1f}")
sys.exit(0 if ok else 1)
