"""Hit model: an impulse J at a world point on a body.  MIRRORED in web/src/hits.ts.

One shared definition of "what a bullet does" is used by training and by the game, so the
policy is trained on exactly the pushes players will give it (PLAN.md §4.4).

An impulse is applied as a constant force over one control step (frame_skip physics steps):
    F = J / (frame_skip * timestep)        so that  F * duration = J
plus the torque from hitting off-center:
    tau = (point - body_com) x F
written into data.xfrc_applied[body] = [F, tau], and cleared after the control step.

Training samples hits (sample_hit); the game builds them from a raycast. Both call apply_hits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from stagger.robot import Robot


@dataclass
class Hit:
    body_id: int
    point: np.ndarray    # world position (3,)
    impulse: np.ndarray  # world impulse J in N*s (3,)


def apply_hits(model: mujoco.MjModel, data: mujoco.MjData, hits: list[Hit], frame_skip: int) -> None:
    """Add each hit's force/torque to xfrc_applied. Call before the control step's mj_steps."""
    duration = frame_skip * model.opt.timestep
    for h in hits:
        force = h.impulse / duration
        torque = np.cross(h.point - data.xipos[h.body_id], force)
        data.xfrc_applied[h.body_id, 0:3] += force
        data.xfrc_applied[h.body_id, 3:6] += torque


def clear_hits(data: mujoco.MjData) -> None:
    """Call after the control step so a hit lasts exactly one control step."""
    data.xfrc_applied[:] = 0.0


# --- Training-side sampling (not mirrored: the game gets hits from raycasts) ----------------


def sample_point_on_body(rng: np.random.Generator, robot: Robot, data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Uniform point inside the body's main geom bounding box, in world coordinates."""
    g = robot.body_main_geom(body_id)
    aabb = robot.model.geom_aabb[g]           # (center xyz, half-size xyz) in the geom frame
    local = aabb[:3] + rng.uniform(-1.0, 1.0, 3) * aabb[3:]
    R = data.geom_xmat[g].reshape(3, 3)
    return data.geom_xpos[g] + R @ local


def sample_direction(rng: np.random.Generator, max_elevation_deg: float) -> np.ndarray:
    """Mostly horizontal unit vector from a random azimuth (shots come from any side)."""
    az = rng.uniform(0.0, 2.0 * math.pi)
    el = math.radians(rng.uniform(-max_elevation_deg, max_elevation_deg))
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def sample_hit(
    rng: np.random.Generator,
    robot: Robot,
    data: mujoco.MjData,
    region_weights: dict[str, float],
    magnitude: float,
    max_elevation_deg: float,
) -> Hit:
    regions = list(region_weights)
    p = np.array([region_weights[r] for r in regions], dtype=float)
    region = regions[rng.choice(len(regions), p=p / p.sum())]
    body_id = int(rng.choice(robot.region_bodies[region]))
    point = sample_point_on_body(rng, robot, data, body_id)
    return Hit(body_id=body_id, point=point, impulse=magnitude * sample_direction(rng, max_elevation_deg))
