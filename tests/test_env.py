"""Sanity checks for the environment.  Run:  python -m tests.test_env   (or: pytest)"""
import numpy as np
from stable_baselines3.common.env_checker import check_env

from sim2real.env import SwingUpEnv


def test_gym_api():
    check_env(SwingUpEnv())
    check_env(SwingUpEnv(prev_action=True))


def test_delay_buffer():
    """With delay=3 the motor must receive 3 zero commands before the first real one."""
    env = SwingUpEnv()
    env.set_params(delay=3)
    env.reset(seed=0)
    applied = []
    for _ in range(6):
        env.step(np.array([1.0], dtype=np.float32))
        applied.append(float(env.data.ctrl[0]))
    assert applied == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_prev_action_in_observation():
    env = SwingUpEnv(prev_action=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,) and obs[3] == 0.0
    obs, *_ = env.step(np.array([0.5], dtype=np.float32))
    assert np.isclose(obs[3], 0.5)


def test_delay_randomization_covers_range():
    env = SwingUpEnv(randomize={"delay": (0, 3)})
    seen = set()
    for seed in range(200):
        env.reset(seed=seed)
        seen.add(env.delay)
    assert seen == {0, 1, 2, 3}


def test_reset_without_randomize_keeps_perturbation():
    env = SwingUpEnv()
    env.set_params(gear=1.3)
    env.reset(seed=0)
    assert np.isclose(env.model.actuator_gear[0, 0], 3 * 1.3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "OK")
