// Debug overlay (~ key): what the balance controller is dealing with, drawn on the floor.
//   cyan disc       center of mass projected on the ground
//   magenta ring    capture point: where the CoM would come to rest (CoM + v / omega0).
//                   When it leaves the support polygon, the robot must step or fall.
//   yellow outline  support polygon (convex hull of feet touching the floor)
//   green feet      foot in contact;  red limbs: motor torque near its limit
//   red arrows      recent hits (length ~ impulse)

import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';
import type { Hit } from './hits';
import type { Scene } from './scene';

const ARROW_SECONDS = 0.6;
const ARROW_M_PER_NS = 0.02;

export class DebugOverlay {
  visible = false;
  private group = new THREE.Group();
  private com: THREE.Mesh;
  private cp: THREE.Mesh;
  private hull: THREE.LineLoop;
  private hullPos = new Float32Array(9 * 3);
  private arrows: { arrow: THREE.ArrowHelper; t: number }[] = [];
  private omega0: number;
  private pelvis: number;
  private floor: number;
  private feet: { geom: number; body: number }[];
  private actBody: number[];
  private actLimit: number[];

  constructor(private mj: MainModule, private model: MjModel, private scene: Scene, standingComHeight: number) {
    this.omega0 = Math.sqrt(9.81 / standingComHeight);
    this.pelvis = model.body('pelvis').id;
    this.floor = model.geom('floor').id;
    this.feet = ['foot_L', 'foot_R'].map((n) => ({ geom: model.geom(n).id, body: model.body(n).id }));
    const trn = model.actuator_trnid, jb = model.jnt_bodyid, fr = model.actuator_forcerange;
    this.actBody = Array.from({ length: model.nu }, (_, i) => jb[trn[i * 2]]);
    this.actLimit = Array.from({ length: model.nu }, (_, i) => Math.max(Math.abs(fr[i * 2]), Math.abs(fr[i * 2 + 1])));

    const flat = (geo: THREE.BufferGeometry, color: number) =>
      new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 }));
    this.com = flat(new THREE.CircleGeometry(0.035, 24), 0x22d3ee);
    this.cp = flat(new THREE.RingGeometry(0.05, 0.07, 32), 0xe879f9);
    const hullGeo = new THREE.BufferGeometry();
    hullGeo.setAttribute('position', new THREE.BufferAttribute(this.hullPos, 3));
    this.hull = new THREE.LineLoop(hullGeo, new THREE.LineBasicMaterial({ color: 0xfacc15, depthTest: false }));
    for (const o of [this.com, this.cp, this.hull]) o.renderOrder = 10;
    this.group.add(this.com, this.cp, this.hull);
    this.group.visible = false;
    scene.scene.add(this.group);
  }

  toggle(): void {
    this.visible = !this.visible;
    this.group.visible = this.visible;
    if (!this.visible) this.scene.setBodyTints(new Map());
  }

  addHits(hits: Hit[]): void {
    if (!this.visible) return;
    for (const h of hits) {
      const J = new THREE.Vector3(...h.impulse);
      const len = Math.max(0.08, J.length() * ARROW_M_PER_NS);
      const dir = J.clone().normalize();
      const origin = new THREE.Vector3(...h.point).addScaledVector(dir, -len);
      const arrow = new THREE.ArrowHelper(dir, origin, len, 0xef4444, Math.min(0.08, len * 0.4), 0.04);
      this.group.add(arrow);
      this.arrows.push({ arrow, t: ARROW_SECONDS });
    }
  }

  update(data: MjData, dt: number): void {
    this.arrows = this.arrows.filter((a) => {
      a.t -= dt;
      if (a.t > 0) return true;
      this.group.remove(a.arrow);
      a.arrow.dispose();
      return false;
    });
    if (!this.visible) return;

    // CoM and capture point (linear inverted pendulum), from the whole-robot subtree.
    this.mj.mj_subtreeVel(this.model, data);
    const c = data.subtree_com, v = data.subtree_linvel, p = this.pelvis;
    const cx = c[p * 3], cy = c[p * 3 + 1];
    this.com.position.set(cx, cy, 0.004);
    this.cp.position.set(cx + v[p * 3] / this.omega0, cy + v[p * 3 + 1] / this.omega0, 0.005);

    // Feet in contact -> support polygon from their box corners.
    const inContact = this.footContacts(data);
    const pts: [number, number][] = [];
    const size = this.model.geom_size, xp = data.geom_xpos, xm = data.geom_xmat;
    this.feet.forEach((f, k) => {
      if (!inContact[k]) return;
      const g = f.geom, sx = size[g * 3], sy = size[g * 3 + 1];
      for (const [a, b] of [[1, 1], [1, -1], [-1, -1], [-1, 1]]) {
        const lx = a * sx, ly = b * sy;
        pts.push([xp[g * 3] + xm[g * 9] * lx + xm[g * 9 + 1] * ly, xp[g * 3 + 1] + xm[g * 9 + 3] * lx + xm[g * 9 + 4] * ly]);
      }
    });
    const hull = convexHull(pts);
    hull.forEach(([x, y], i) => this.hullPos.set([x, y, 0.006], i * 3));
    this.hull.geometry.setDrawRange(0, hull.length);
    this.hull.geometry.attributes.position.needsUpdate = true;

    // Tints: torque load per body (max |tau| / limit over its motors), feet green when planted.
    const tints = new Map<number, THREE.Color>();
    const force = data.actuator_force;
    for (let i = 0; i < this.actBody.length; i++) {
      const load = Math.min(1, Math.abs(force[i]) / this.actLimit[i]);
      const b = this.actBody[i];
      const prev = tints.get(b);
      if (load > 0.3 && (!prev || prev.r < load)) tints.set(b, new THREE.Color(load * 0.9, 0, 0));
    }
    this.feet.forEach((f, k) => inContact[k] && tints.set(f.body, new THREE.Color(0, 0.45, 0.1)));
    this.scene.setBodyTints(tints);
  }

  private footContacts(data: MjData): boolean[] {
    const out = this.feet.map(() => false);
    const n = data.ncon;
    if (!n) return out;
    const contacts = data.contact;
    try {
      for (let i = 0; i < n; i++) {
        const ct = contacts.get(i)!;
        const g1 = ct.geom1, g2 = ct.geom2;
        ct.delete();
        this.feet.forEach((f, k) => {
          if ((g1 === f.geom && g2 === this.floor) || (g2 === f.geom && g1 === this.floor)) out[k] = true;
        });
      }
    } finally {
      contacts.delete();
    }
    return out;
  }
}

/** 2D convex hull (monotone chain), counter-clockwise. */
function convexHull(points: [number, number][]): [number, number][] {
  if (points.length < 3) return points;
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: [number, number][] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: [number, number][] = [];
  for (const q of p.reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}
