// Hit model. MIRROR of stagger/hits.py (apply_hits / clear_hits) -- keep behavior identical.
//
// A hit is an impulse J (N*s) at a world point on a body, applied as a constant force for one
// control step (frame_skip physics steps):  F = J / (frameSkip * timestep),
// plus the off-center torque  tau = (point - body_com) x F,  written to xfrc_applied[body].
// The game builds hits from raycasts; training samples them. Same physics either way.

import type { MjData, MjModel } from '@mujoco/mujoco';

export type Vec3 = [number, number, number];

export interface Hit {
  bodyId: number;
  point: Vec3;   // world position
  impulse: Vec3; // world impulse, N*s
}

/** Add each hit's force/torque to xfrc_applied. Call right before the control step's mj_steps. */
export function applyHits(model: MjModel, data: MjData, hits: Hit[], frameSkip: number): void {
  const duration = frameSkip * model.opt.timestep;
  const xipos = data.xipos;        // live views into WASM memory: re-read, never cache across steps
  const xfrc = data.xfrc_applied;
  for (const h of hits) {
    const b = h.bodyId;
    const fx = h.impulse[0] / duration, fy = h.impulse[1] / duration, fz = h.impulse[2] / duration;
    const rx = h.point[0] - xipos[b * 3], ry = h.point[1] - xipos[b * 3 + 1], rz = h.point[2] - xipos[b * 3 + 2];
    xfrc[b * 6 + 0] += fx;
    xfrc[b * 6 + 1] += fy;
    xfrc[b * 6 + 2] += fz;
    xfrc[b * 6 + 3] += ry * fz - rz * fy;
    xfrc[b * 6 + 4] += rz * fx - rx * fz;
    xfrc[b * 6 + 5] += rx * fy - ry * fx;
  }
}

/** Call after the control step so a hit lasts exactly one control step. */
export function clearHits(data: MjData): void {
  data.xfrc_applied.fill(0);
}
