"""Single-pendulum swing-up environment (MuJoCo + Gymnasium) with a perturbable simulator.

The environment exposes the "sim-to-real gap" as parameters that can be changed:
motor gain (``gear``), mass, joint damping and action latency (``delay``).
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from .model import XML


class SwingUpEnv(gym.Env):
    """Swing a pendulum up from its hanging position using a single motor at the pivot.

    theta = 0 is the hanging position, theta = pi the upright one.

    Observation: [cos(theta), sin(theta), omega] (+ previous action if ``prev_action``).
    Action:      motor command in [-1, 1].
    Reward:      -cos(theta) - 0.01*omega^2 - 0.001*a^2 - 0.001*(a_prev - a)^2.
    Episode:     ``max_steps`` control steps (default 250 x 20 ms = 5 s), no early termination.

    Args:
        frame_skip: physics steps (2 ms each) per control step.
        max_steps: episode length in control steps.
        randomize: dict {param: (lo, hi)} used for domain randomization at every reset().
            Supported params: "gear", "mass", "damping" (multiplicative factors of the
            nominal value, drawn uniformly) and "delay" (integer number of control steps,
            drawn uniformly in [lo, hi]). Empty/None -> the physics is never touched.
        prev_action: if True, the previous commanded action is appended to the observation.
    """

    def __init__(self, frame_skip=10, max_steps=250, randomize=None, prev_action=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(XML)
        m = self.model
        # Nominal values are stored once: every perturbation is a factor of these.
        self._nom = dict(mass=m.body_mass.copy(), inertia=m.body_inertia.copy(),
                         damping=m.dof_damping.copy(), gear=m.actuator_gear.copy())
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.randomize = randomize or {}
        self.prev_action = prev_action

        self.t = 0
        self.action_p = 0.0      # previous commanded action
        self.delay = 0           # action latency, in control steps
        self.buf = []            # FIFO of commands waiting to be applied

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(4 if prev_action else 3,), dtype=np.float32)

    def _get_obs(self):
        theta = self.data.qpos[0]
        omega = self.data.qvel[0]
        obs = [np.cos(theta), np.sin(theta), omega]
        if self.prev_action:
            obs.append(self.action_p)
        return np.array(obs, dtype=np.float32)

    def set_params(self, mass=1.0, damping=1.0, gear=1.0, delay=0):
        """Set the physics to nominal * factor (delay is an absolute number of control steps)."""
        m = self.model
        m.body_mass[:] = self._nom["mass"] * mass
        m.body_inertia[:] = self._nom["inertia"] * mass   # inertia follows the mass
        m.dof_damping[:] = self._nom["damping"] * damping
        m.actuator_gear[:] = self._nom["gear"] * gear
        self.delay = int(delay)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize:
            p = {"mass": 1.0, "damping": 1.0, "gear": 1.0, "delay": 0}
            for k, (lo, hi) in self.randomize.items():
                if k == "delay":
                    p[k] = self.np_random.integers(lo, hi + 1)
                else:
                    p[k] = self.np_random.uniform(lo, hi)
            self.set_params(**p)

        self.buf = [0.0] * self.delay                  # `delay` zero commands before the first real one
        mujoco.mj_resetData(self.model, self.data)     # must come first
        self.data.qpos[0] = self.np_random.uniform(-0.1, 0.1)
        mujoco.mj_forward(self.model, self.data)       # recompute derived quantities after writing qpos
        self.t = 0
        self.action_p = 0.0
        return self._get_obs(), {}

    def step(self, action):
        # Latency: the command applied now is the one issued `delay` steps ago.
        self.buf.append(float(action[0]))
        self.data.ctrl[0] = self.buf.pop(0)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.t += 1

        theta = self.data.qpos[0]
        omega = self.data.qvel[0]
        a = float(action[0])
        reward = (-np.cos(theta) - 0.01 * omega ** 2
                  - 0.001 * a ** 2 - 0.001 * (self.action_p - a) ** 2)
        self.action_p = a

        terminated = False                       # no failure state
        truncated = self.t >= self.max_steps     # time limit
        return self._get_obs(), reward, terminated, truncated, {}
