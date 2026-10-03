// Observation builder. MIRROR of stagger/obs.py -- keep behavior identical (PLAN.md §7).
//
// Pure function of qpos, qvel and the previous action (no derived MjData fields), so it
// gives the same numbers as training no matter when it is called.
//
// Layout (67 floats):
//   [0:3]   base angular velocity, body frame      * ang_vel
//   [3:6]   gravity direction, body frame          * gravity
//   [6:9]   base linear velocity, body frame       * lin_vel
//   [9:11]  offset to home spot, heading frame     * home
//   [11:13] heading error (sin, cos)
//   [13:31] joint pos - default pose               * joint_pos
//   [31:49] joint velocities                       * joint_vel
//   [49:67] previous action

export const OBS_DIM = 67;
export const NU = 18;

export interface ObsScales {
  ang_vel: number;
  gravity: number;
  lin_vel: number;
  home: number;
  joint_pos: number;
  joint_vel: number;
}

type Vec = ArrayLike<number>;

/** Unit quaternion (w, x, y, z) at q[o..o+3] -> row-major 3x3 rotation (body -> world). */
export function quatToMat(q: Vec, o = 0): number[] {
  const w = q[o], x = q[o + 1], y = q[o + 2], z = q[o + 3];
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

/** Heading of the body's forward (x) axis projected onto the floor. */
export function yawFromMat(R: number[]): number {
  return Math.atan2(R[3], R[0]);
}

export function wrapAngle(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

export function buildObs(
  qpos: Vec,
  qvel: Vec,
  prevAction: Vec,
  defaultJointPos: Vec,
  homeXY: Vec,
  homeYaw: number,
  s: ObsScales,
  clip: number,
  out: Float64Array = new Float64Array(OBS_DIM),
): Float64Array {
  const R = quatToMat(qpos, 3);
  const yaw = yawFromMat(R);
  const c = Math.cos(yaw), sn = Math.sin(yaw);

  // angular velocity: freejoint qvel[3:6] is already in the body frame
  out[0] = qvel[3] * s.ang_vel;
  out[1] = qvel[4] * s.ang_vel;
  out[2] = qvel[5] * s.ang_vel;
  // gravity in body frame = R^T (0, 0, -1) = -(third row of R)
  out[3] = -R[6] * s.gravity;
  out[4] = -R[7] * s.gravity;
  out[5] = -R[8] * s.gravity;
  // linear velocity: qvel[0:3] is world frame -> R^T v
  const vx = qvel[0], vy = qvel[1], vz = qvel[2];
  out[6] = (R[0] * vx + R[3] * vy + R[6] * vz) * s.lin_vel;
  out[7] = (R[1] * vx + R[4] * vy + R[7] * vz) * s.lin_vel;
  out[8] = (R[2] * vx + R[5] * vy + R[8] * vz) * s.lin_vel;
  // offset to home, rotated into the heading frame
  const dx = homeXY[0] - qpos[0], dy = homeXY[1] - qpos[1];
  out[9] = (c * dx + sn * dy) * s.home;
  out[10] = (-sn * dx + c * dy) * s.home;
  const err = wrapAngle(yaw - homeYaw);
  out[11] = Math.sin(err);
  out[12] = Math.cos(err);
  for (let i = 0; i < NU; i++) {
    out[13 + i] = (qpos[7 + i] - defaultJointPos[i]) * s.joint_pos;
    out[31 + i] = qvel[6 + i] * s.joint_vel;
    out[49 + i] = prevAction[i];
  }
  for (let i = 0; i < OBS_DIM; i++) out[i] = Math.min(clip, Math.max(-clip, out[i]));
  return out;
}
