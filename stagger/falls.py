"""Knockdown detection.  MIRRORED in web/src/falls.ts.

Knockdown = (PLAN.md §4.5)
  - any "ground body" (head, torso, pelvis, arms, thighs) touching the floor, or
  - pelvis height < min_frac * standing height, or
  - torso tilt > max_tilt from vertical.
It ends a training episode and is a KO in the game.

Tilt and height come from qpos (always current). Floor contacts come from data.contact,
which mj_step fills at the start of its last substep -- the same on both sides.
"""

from __future__ import annotations

import math

import mujoco

from stagger.config import FallsCfg
from stagger.robot import Robot

OK = ""


class FallDetector:
    def __init__(self, robot: Robot, cfg: FallsCfg):
        self.floor = robot.floor_geom
        # geom id -> body name for every geom that must not touch the floor
        self.ground_geoms = {
            g: robot.model.body(robot.model.geom_bodyid[g]).name for g in robot.geoms_of_bodies(cfg.ground_bodies)
        }
        self.min_height = cfg.min_pelvis_height_frac * robot.standing_height
        self.min_up = math.cos(math.radians(cfg.max_torso_tilt_deg))

    def check(self, data: mujoco.MjData) -> str:
        """Return "" if still standing, otherwise the knockdown reason (for logs and the HUD)."""
        for i in range(data.ncon):
            g1, g2 = data.contact.geom[i]
            other = g2 if g1 == self.floor else g1 if g2 == self.floor else -1
            if other in self.ground_geoms:
                return f"contact:{self.ground_geoms[other]}"
        if data.qpos[2] < self.min_height:
            return "height"
        # Torso is welded to the pelvis, so the pelvis quaternion gives the torso tilt.
        # up = world z of the body z axis = R[2,2] = 1 - 2(x^2 + y^2).
        _, x, y, _ = data.qpos[3:7]
        if 1.0 - 2.0 * (x * x + y * y) < self.min_up:
            return "tilt"
        return OK

