"""Domain randomization (PLAN.md §4.8).

Each episode gets slightly different physics: floor friction, body masses, motor stiffness
and torque limits, observation noise and 0-1 steps of action latency. A policy that only
works for one exact set of numbers is brittle; one that works across a range is robust to
the things we can't match perfectly (and to browser frame-timing jitter in the game).

Each env process owns its own MjModel, so mutating it here only affects that env.
"""

from __future__ import annotations

import mujoco
import numpy as np

from stagger.config import RandomizationCfg


class Randomizer:
    def __init__(self, model: mujoco.MjModel, cfg: RandomizationCfg):
        self.model = model
        self.cfg = cfg
        # Nominal values, so every episode samples around the XML values, not the last sample.
        self.friction0 = model.geom_friction[:, 0].copy()
        self.mass0 = model.body_mass.copy()
        self.inertia0 = model.body_inertia.copy()
        self.kp0 = model.actuator_gainprm[:, 0].copy()
        self.forcerange0 = model.actuator_forcerange.copy()
        # Defaults when disabled.
        self.obs_noise_std = 0.0
        self.latency_steps = 0

    def reset(self, rng: np.random.Generator) -> None:
        m, c = self.model, self.cfg
        if not c.enabled:
            return

        # Friction: one value for the floor and every robot geom (MuJoCo uses the max of a pair).
        m.geom_friction[:, 0] = rng.uniform(*c.friction)

        # Mass: scale each body's mass and inertia together (same shape, different density).
        s = rng.uniform(*c.mass_scale, size=m.nbody)
        s[0] = 1.0  # world body
        m.body_mass[:] = self.mass0 * s
        m.body_inertia[:] = self.inertia0 * s[:, None]

        # Motors: position servo stiffness kp lives in gainprm[0] and -kp in biasprm[1].
        kp = self.kp0 * rng.uniform(*c.kp_scale, size=m.nu)
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.actuator_forcerange[:] = self.forcerange0 * rng.uniform(*c.force_limit_scale, size=m.nu)[:, None]

        self.obs_noise_std = c.obs_noise_std
        self.latency_steps = int(rng.integers(c.action_latency_steps[0], c.action_latency_steps[1] + 1))
