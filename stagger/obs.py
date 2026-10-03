"""Observation builder.  MIRRORED in web/src/obs.ts -- keep behavior identical (PLAN.md §7).

The observation is a pure function of qpos, qvel and the previous action. It deliberately
uses no derived MjData fields (xmat, cvel, ...): those are only refreshed by mj_forward,
so reading them right after mj_step would be one physics substep stale, and the browser
would have to replicate that exact staleness. qpos/qvel are always current.

The policy never sees the bullets. It only feels a hit through its own body: a sudden
change in angular/linear velocity, tilt and joint state (PLAN.md §4.3).

Layout (67 floats):
  [0:3]   base angular velocity, body frame      * ang_vel
  [3:6]   gravity direction, body frame          * gravity   ("which way is down")
  [6:9]   base linear velocity, body frame       * lin_vel
  [9:11]  offset to home spot, heading frame     * home      (lets it step back)
  [11:13] heading error (sin, cos)                            (lets it turn back)
  [13:31] joint pos - default pose               * joint_pos
  [31:49] joint velocities                       * joint_vel
  [49:67] previous action                                     (already in [-1, 1])
"""

from __future__ import annotations

import math

import numpy as np

OBS_DIM = 67
NU = 18


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix (body frame -> world frame)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_from_mat(R: np.ndarray) -> float:
    """Heading angle of the body's forward (x) axis projected onto the floor."""
    return math.atan2(R[1, 0], R[0, 0])


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def build_obs(
    qpos: np.ndarray,
    qvel: np.ndarray,
    prev_action: np.ndarray,
    default_joint_pos: np.ndarray,
    home_xy: np.ndarray,
    home_yaw: float,
    scales: dict[str, float],
    clip: float,
) -> np.ndarray:
    R = quat_to_mat(qpos[3:7])
    yaw = yaw_from_mat(R)
    c, s = math.cos(yaw), math.sin(yaw)

    ang_vel = qvel[3:6]                   # freejoint angular velocity is already in body frame
    gravity = -R[2, :]                    # R^T @ (0, 0, -1)
    lin_vel = R.T @ qvel[0:3]             # freejoint linear velocity is in world frame
    dx, dy = home_xy[0] - qpos[0], home_xy[1] - qpos[1]
    home = np.array([c * dx + s * dy, -s * dx + c * dy])  # rotate world offset into heading frame
    err = wrap_angle(yaw - home_yaw)

    obs = np.concatenate(
        [
            ang_vel * scales["ang_vel"],
            gravity * scales["gravity"],
            lin_vel * scales["lin_vel"],
            home * scales["home"],
            [math.sin(err), math.cos(err)],
            (qpos[7:] - default_joint_pos) * scales["joint_pos"],
            qvel[6:] * scales["joint_vel"],
            prev_action,
        ]
    )
    return np.clip(obs, -clip, clip).astype(np.float32)
