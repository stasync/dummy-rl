// Knockdown detection. MIRROR of stagger/falls.py -- keep behavior identical.
//
// Knockdown = a "ground body" (head, torso, pelvis, arms, thighs) touching the floor, or
// pelvis height < frac * standing height, or torso tilt > max tilt. Checked in that order,
// and contacts in MuJoCo's contact order, so the reported reason matches Python exactly.

import type { MjData, MjModel } from '@mujoco/mujoco';

export interface FallsConfig {
  ground_bodies: string[];
  min_pelvis_height_frac: number;
  max_torso_tilt_deg: number;
}

export class FallDetector {
  private floor: number;
  private groundGeoms = new Map<number, string>(); // geom id -> body name
  private minHeight: number;
  private minUp: number;

  constructor(model: MjModel, cfg: FallsConfig, standingHeight: number) {
    this.floor = model.geom('floor').id;
    const bodyNames = new Map<number, string>(cfg.ground_bodies.map((n) => [model.body(n).id, n]));
    const geomBody = model.geom_bodyid;
    for (let g = 0; g < model.ngeom; g++) {
      const name = bodyNames.get(geomBody[g]);
      if (name !== undefined) this.groundGeoms.set(g, name);
    }
    this.minHeight = cfg.min_pelvis_height_frac * standingHeight;
    this.minUp = Math.cos((cfg.max_torso_tilt_deg * Math.PI) / 180);
  }

  /** "" if still standing, otherwise the reason ("contact:<body>", "height" or "tilt"). */
  check(data: MjData): string {
    const ncon = data.ncon;
    if (ncon > 0) {
      const contacts = data.contact; // a copy: must be deleted
      try {
        for (let i = 0; i < ncon; i++) {
          const c = contacts.get(i)!;
          const g1 = c.geom1, g2 = c.geom2;
          c.delete();
          const other = g1 === this.floor ? g2 : g2 === this.floor ? g1 : -1;
          const name = this.groundGeoms.get(other);
          if (name !== undefined) return `contact:${name}`;
        }
      } finally {
        contacts.delete();
      }
    }
    const qpos = data.qpos;
    if (qpos[2] < this.minHeight) return 'height';
    // Torso is welded to the pelvis: up = world z of body z axis = 1 - 2(x^2 + y^2).
    const x = qpos[4], y = qpos[5];
    if (1 - 2 * (x * x + y * y) < this.minUp) return 'tilt';
    return '';
  }
}
