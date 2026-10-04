// "Brain" debug panel (Tab key or the HUD's debug button): what the neural network sees, thinks and does, and how that
// becomes motor torque on the robot. Drawn on a 2D canvas, docked on the right.
//
//   SEES       the 67 observation values (blue = negative, orange = positive)
//   THINKS     the actor's hidden layers (256, 256, 128 neurons), activation per neuron
//   DOES       per joint: network output a -> PD target (default + scale * a) vs actual angle -> torque
//   REACTING   gradient x input attribution per input group: what is driving the output right now
//   TIMELINE   last 5 s: balance confidence V(s), network effort, motor load; ticks = hits
//
// The ghost robot (scene.ts) shows the same PD targets as a pose.

import type { MainModule, MjModel } from '@mujoco/mujoco';
import type { Sim } from './sim';

const W = 360;          // CSS px
const PAD = 10;
const CW = W - 2 * PAD; // content width
const HISTORY = 250;    // control steps = 5 s at 50 Hz
const ATTRIB_EVERY_S = 0.1;

const C = {
  bg: 'rgba(17,18,20,0.86)',
  cell: [42, 44, 48],
  neg: [59, 130, 246],
  pos: [240, 115, 24],
  text: '#ebe8e2',
  muted: '#8b8f96',
  target: '#38bdf8', // = ghost colour
  ok: '#4ade80',
  warn: '#f07318',
  bad: '#ef4444',
};

const OBS_GROUPS: [string, number][] = [
  ['spin', 3], ['tilt', 3], ['velocity', 3], ['home', 2], ['heading', 2],
  ['joint angles', 18], ['joint speeds', 18], ['last action', 18],
];

interface JointInfo {
  act: number;    // actuator index (= action index)
  label: string;  // "hip pitch"
  lo: number;     // joint range, rad
  hi: number;
  qadr: number;   // qpos address
  tauMax: number; // torque limit, N*m
  kp: number;     // PD stiffness, N*m/rad
}

/** Diverging colour for v in [-1, 1]: blue (neg) / dark (0) / orange (pos). */
function diverge(v: number): string {
  const t = Math.min(1, Math.abs(v));
  const to = v < 0 ? C.neg : C.pos;
  const ch = (i: number) => Math.round(C.cell[i] + (to[i] - C.cell[i]) * t);
  return `rgb(${ch(0)},${ch(1)},${ch(2)})`;
}

const loadColor = (x: number) => (x < 0.5 ? C.ok : x < 0.85 ? C.warn : C.bad);

export class BrainPanel {
  private canvas: HTMLCanvasElement;
  private g: CanvasRenderingContext2D;
  private limbs: { title: string; joints: JointInfo[] }[];
  private conf = new Float32Array(HISTORY);
  private effort = new Float32Array(HISTORY);
  private load = new Float32Array(HISTORY);
  private hit = new Float32Array(HISTORY);
  private head = 0;
  private attrib = new Float64Array(67);
  private groupShare = new Float64Array(OBS_GROUPS.length);
  private attribTimer = 0;
  private height = 0;

  constructor(mj: MainModule, model: MjModel) {
    this.canvas = document.getElementById('brain') as HTMLCanvasElement;
    this.g = this.canvas.getContext('2d')!;

    const ACT = mj.mjtObj.mjOBJ_ACTUATOR.value;
    const gainStride = model.actuator_gainprm.length / model.nu;
    const joints: JointInfo[] = [];
    for (let i = 0; i < model.nu; i++) {
      const jid = model.actuator_trnid[i * 2];
      const name = mj.mj_id2name(model, ACT, i); // e.g. "hip_pitch_L"
      joints.push({
        act: i,
        label: name.slice(0, -2).replace('_', ' '),
        lo: model.jnt_range[jid * 2],
        hi: model.jnt_range[jid * 2 + 1],
        qadr: model.jnt_qposadr[jid],
        tauMax: model.actuator_forcerange[i * 2 + 1],
        kp: model.actuator_gainprm[i * gainStride],
      });
    }
    const side = (s: string) => joints.filter((j) => mj.mj_id2name(model, ACT, j.act).endsWith(s));
    const legs = (s: string) => side(s).filter((j) => /hip|knee|ankle/.test(j.label));
    const arms = (s: string) => side(s).filter((j) => !/hip|knee|ankle/.test(j.label));
    this.limbs = [
      { title: 'LEFT LEG', joints: legs('_L') },
      { title: 'RIGHT LEG', joints: legs('_R') },
      { title: 'LEFT ARM', joints: arms('_L') },
      { title: 'RIGHT ARM', joints: arms('_R') },
    ];
    this.setVisible(false);
  }

  setVisible(on: boolean): void {
    this.canvas.hidden = !on;
  }

  /** Call once per control step (50 Hz) to fill the timeline. */
  record(sim: Sim): void {
    const d = sim.data;
    let effort = 0, load = 0, n = 0;
    for (const limb of this.limbs) {
      for (const j of limb.joints) {
        effort += sim.aiAction[j.act] ** 2;
        load += Math.min(1, Math.abs(d.actuator_force[j.act]) / j.tauMax);
        n++;
      }
    }
    const i = this.head;
    this.conf[i] = sim.policy.confidence(sim.value);
    this.effort[i] = Math.sqrt(effort / n);
    this.load[i] = load / n;
    this.hit[i] = sim.lastHits.reduce((s, h) => s + Math.hypot(...h.impulse), 0);
    this.head = (i + 1) % HISTORY;
  }

  draw(sim: Sim, dt: number): void {
    if (this.canvas.hidden) return;
    this.updateAttribution(sim, dt);
    this.resize();
    const g = this.g;
    g.clearRect(0, 0, W, this.height);
    g.fillStyle = C.bg;
    g.fillRect(0, 0, W, this.height);

    let y = PAD;
    const applied = sim.mode === 'ai' && sim.powered;
    y = this.title(y, 'NEURAL CONTROLLER', applied ? 'output drives the motors' : `${sim.mode === 'limp' || !sim.powered ? 'motors off' : 'Stiff'}: output shown, not applied`);
    y = this.drawInputs(y + 4, sim.obs);
    y = this.drawHidden(y + 8, sim.policy.actorActivations());
    y = this.drawJoints(y + 8, sim, applied);
    y = this.drawAttribution(y + 8);
    y = this.drawTimeline(y + 8);
    const needed = Math.ceil(y + PAD);
    if (needed !== this.height) {
      this.height = needed;
      this.resize(true);
    }
  }

  // --- sections -----------------------------------------------------------------------------

  private title(y: number, text: string, sub: string): number {
    const g = this.g;
    g.font = '700 11px ui-monospace, Menlo, monospace';
    g.fillStyle = C.warn;
    g.textBaseline = 'top';
    g.fillText(text, PAD, y);
    g.font = '10px ui-monospace, Menlo, monospace';
    g.fillStyle = C.muted;
    g.fillText(sub, PAD + 132, y + 1);
    return y + 14;
  }

  private label(x: number, y: number, text: string, color = C.muted, size = 9): void {
    this.g.font = `${size}px ui-monospace, Menlo, monospace`;
    this.g.fillStyle = color;
    this.g.textBaseline = 'top';
    this.g.fillText(text, x, y);
  }

  private drawInputs(y: number, obs: Float64Array): number {
    this.label(PAD, y, 'SEES  67 inputs (its own body only: it never sees the bullets)', C.text, 9);
    y += 12;
    const cw = CW / obs.length, g = this.g;
    for (let i = 0; i < obs.length; i++) {
      g.fillStyle = diverge(Math.tanh(obs[i] * 2));
      g.fillRect(PAD + i * cw, y, Math.max(1, cw - 0.6), 12);
    }
    let x = 0;
    g.fillStyle = C.muted;
    for (const [name, n] of OBS_GROUPS) {
      g.fillRect(PAD + x * cw - 0.5, y - 2, 1, 16);
      const w = n * cw;
      if (w > 22) this.label(PAD + x * cw + 2, y + 15, name);
      x += n;
    }
    return y + 26;
  }

  private drawHidden(y: number, layers: Float64Array[]): number {
    this.label(PAD, y, 'THINKS  hidden layers (each cell = one neuron)', C.text, 9);
    y += 12;
    const g = this.g, perRow = 128, cw = CW / perRow;
    for (let li = 0; li < layers.length - 1; li++) {
      const a = layers[li];
      const rows = Math.ceil(a.length / perRow);
      for (let i = 0; i < a.length; i++) {
        g.fillStyle = diverge(Math.tanh(a[i]));
        g.fillRect(PAD + (i % perRow) * cw, y + Math.floor(i / perRow) * 6, Math.max(1, cw - 0.4), 5);
      }
      y += rows * 6 + 3;
    }
    return y;
  }

  private drawJoints(y: number, sim: Sim, applied: boolean): number {
    const d = sim.data, c = sim.policy.contract, g = this.g;
    const X = { name: PAD, act: PAD + 78, angle: PAD + 152, torque: PAD + 270 };
    const Wd = { act: 66, angle: 112, torque: CW - 270 };
    this.label(PAD, y, 'DOES  network output → PD target → torque', C.text, 9);
    y += 12;
    this.label(X.act, y, 'network a');
    this.label(X.angle, y, 'angle: actual', C.text);
    this.label(X.angle + 74, y, '→target', C.target);
    this.label(X.torque, y, 'torque');
    y += 11;

    for (const limb of this.limbs) {
      this.label(PAD, y + 1, limb.title, C.muted, 8);
      y += 10;
      for (const j of limb.joints) {
        const a = sim.aiAction[j.act];
        this.label(X.name + 4, y + 1, j.label, C.text);

        // network output, -1..1 (dimmed when not applied)
        g.fillStyle = 'rgba(255,255,255,0.07)';
        g.fillRect(X.act, y + 2, Wd.act, 8);
        g.globalAlpha = applied ? 1 : 0.35;
        g.fillStyle = diverge(a);
        const mid = X.act + Wd.act / 2;
        g.fillRect(Math.min(mid, mid + (a * Wd.act) / 2), y + 2, Math.abs(a * Wd.act) / 2, 8);
        g.globalAlpha = 1;
        g.fillStyle = C.muted;
        g.fillRect(mid - 0.5, y + 1, 1, 10);

        // angle within the joint range: default (grey), actual (white), target (cyan), pull (line)
        const toX = (q: number) => X.angle + ((q - j.lo) / (j.hi - j.lo)) * Wd.angle;
        const q = d.qpos[j.qadr];
        const target = applied ? d.ctrl[j.act] : c.default_joint_pos[j.act];
        g.fillStyle = 'rgba(255,255,255,0.07)';
        g.fillRect(X.angle, y + 2, Wd.angle, 8);
        g.fillStyle = C.muted;
        g.fillRect(toX(c.default_joint_pos[j.act]) - 0.5, y + 1, 1, 10);
        g.fillStyle = 'rgba(240,115,24,0.55)';
        g.fillRect(Math.min(toX(q), toX(target)), y + 5, Math.abs(toX(target) - toX(q)), 2);
        g.fillStyle = C.target;
        g.fillRect(toX(target) - 1, y + 1, 2, 10);
        g.fillStyle = C.text;
        g.fillRect(toX(q) - 1, y + 2, 2, 8);

        // torque vs limit, signed
        const tau = d.actuator_force[j.act], r = Math.max(-1, Math.min(1, tau / j.tauMax));
        const tmid = X.torque + Wd.torque / 2;
        g.fillStyle = 'rgba(255,255,255,0.07)';
        g.fillRect(X.torque, y + 2, Wd.torque, 8);
        g.fillStyle = loadColor(Math.abs(r));
        g.fillRect(Math.min(tmid, tmid + (r * Wd.torque) / 2), y + 2, Math.abs(r * Wd.torque) / 2, 8);
        g.fillStyle = C.muted;
        g.fillRect(tmid - 0.5, y + 1, 1, 10);
        y += 12;
      }
    }
    const kps = [...new Set(this.limbs.flatMap((l) => l.joints.map((j) => `${j.kp}`)))].join('/');
    const taus = [...new Set(this.limbs.flatMap((l) => l.joints.map((j) => `${j.tauMax}`)))].join('/');
    this.label(PAD, y + 2, `target = default + ${c.action_scale} rad × a · kp ${kps} N·m/rad · limits ±${taus} N·m`);
    return y + 14;
  }

  private updateAttribution(sim: Sim, dt: number): void {
    this.attribTimer -= dt;
    if (this.attribTimer > 0) return;
    this.attribTimer = ATTRIB_EVERY_S;
    sim.policy.attribution(sim.obs, this.attrib); // actor activations are from this obs (last control step)
    let total = 0, o = 0;
    const sums = OBS_GROUPS.map(([, n]) => {
      let s = 0;
      for (let i = 0; i < n; i++) s += this.attrib[o + i];
      o += n;
      total += s;
      return s;
    });
    sums.forEach((s, k) => (this.groupShare[k] += 0.4 * ((total > 0 ? s / total : 0) - this.groupShare[k])));
  }

  private drawAttribution(y: number): number {
    this.label(PAD, y, 'REACTING TO  share of |∂output/∂input × input|', C.text, 9);
    y += 12;
    const g = this.g, bx = PAD + 82, bw = CW - 82 - 34;
    OBS_GROUPS.forEach(([name, ], k) => {
      const share = this.groupShare[k];
      this.label(PAD + 4, y, name, C.text);
      g.fillStyle = 'rgba(255,255,255,0.07)';
      g.fillRect(bx, y + 1, bw, 7);
      g.fillStyle = C.warn;
      g.fillRect(bx, y + 1, bw * share, 7);
      this.label(bx + bw + 4, y, `${Math.round(share * 100)}%`);
      y += 10;
    });
    return y;
  }

  private drawTimeline(y: number): number {
    this.label(PAD, y, 'TIMELINE  last 5 s (red ticks = hits)', C.text, 9);
    y += 12;
    const g = this.g, h = 24;
    const series: [string, Float32Array, string][] = [
      ['balance V(s)', this.conf, C.ok],
      ['network effort', this.effort, C.warn],
      ['motor load', this.load, C.bad],
    ];
    const x0 = PAD + 90, w = CW - 90;
    const top = y;
    for (const [name, arr, color] of series) {
      this.label(PAD + 4, y + 7, name, C.text);
      g.fillStyle = 'rgba(255,255,255,0.05)';
      g.fillRect(x0, y, w, h);
      g.strokeStyle = color;
      g.lineWidth = 1.2;
      g.beginPath();
      for (let k = 0; k < HISTORY; k++) {
        const v = arr[(this.head + k) % HISTORY];
        const px = x0 + (k / (HISTORY - 1)) * w, py = y + h - Math.min(1, v) * (h - 2) - 1;
        if (k === 0) g.moveTo(px, py);
        else g.lineTo(px, py);
      }
      g.stroke();
      y += h + 4;
    }
    g.fillStyle = 'rgba(239,68,68,0.8)';
    for (let k = 0; k < HISTORY; k++) {
      if (this.hit[(this.head + k) % HISTORY] > 0) g.fillRect(x0 + (k / (HISTORY - 1)) * w - 0.5, top, 1, y - top - 4);
    }
    return y;
  }

  private resize(force = false): void {
    const dpr = Math.min(window.devicePixelRatio, 2);
    if (!force && this.canvas.width === Math.round(W * dpr) && this.height) return;
    const h = this.height || 600;
    this.canvas.width = Math.round(W * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${h}px`;
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}
