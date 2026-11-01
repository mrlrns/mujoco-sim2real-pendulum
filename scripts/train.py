"""Train one PPO policy.

    python -m scripts.train --family base --seed 0 --out models/base_s0
    python -m scripts.train --family dr   --seed 0 --out models/dr_s0
    python -m scripts.train --family lat  --lat-max 5 --seed 0 --out models/lat5_s0
    python -m scripts.train --family full --seed 0 --out models/full_s0
    python -m scripts.train --family base --prev --seed 0 --out models/apbase_s0   # obs + previous action

Families (what is randomized at every episode):
    base : nothing (nominal physics)
    dr   : gear x U(0.7, 1.3), mass x U(0.8, 1.2)
    lat  : action delay ~ U{0..lat-max} control steps (1 step = 20 ms)
    full : gear, mass and delay in {0..3}
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from sim2real.env import SwingUpEnv

RANGES = {
    "base": None,
    "dr": {"gear": (0.7, 1.3), "mass": (0.8, 1.2)},
    "full": {"gear": (0.7, 1.3), "mass": (0.8, 1.2), "delay": (0, 3)},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["base", "dr", "lat", "full"], default="base")
    ap.add_argument("--lat-max", type=int, default=5, help="max delay (control steps) for --family lat")
    ap.add_argument("--prev", action="store_true", help="append the previous action to the observation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--out", required=True, help="model path without extension, e.g. models/dr_s0")
    a = ap.parse_args()

    if a.family == "lat":
        rnd = {"delay": (0, a.lat_max)}
        name = f"lat{a.lat_max}"
    else:
        rnd = RANGES[a.family]
        name = a.family
    tag = ("ap_" if a.prev else "") + name

    env = make_vec_env(lambda: Monitor(SwingUpEnv(randomize=rnd, prev_action=a.prev)),
                       n_envs=8, seed=a.seed)
    model = PPO("MlpPolicy", env, seed=a.seed, device="cpu", verbose=0,
                n_steps=256, batch_size=256, gamma=0.98,
                tensorboard_log=f"runs/{tag}_s{a.seed}")
    model.learn(a.steps)
    model.save(a.out)
    print(a.out, "done")
