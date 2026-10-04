// The live simulation: MuJoCo WASM physics + the trained policy, stepped exactly like training.
//
// One control step (50 Hz) = build obs -> policy action -> PD targets -> apply pending hits
// -> frame_skip physics steps (500 Hz) -> clear hits -> knockdown check.
// That is the same order as StaggerEnv.step, which is what the parity test guarantees.

import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';
import { FallDetector } from './falls';
import { applyHits, clearHits, type Hit } from './hits';
import { buildObs, OBS_DIM } from './obs';
import type { Policy } from './policy';

export type Mode = 'ai' | 'stiff' | 'limp';

/** Everything needed to put the simulation back exactly where it was (rewind). */
export interface SimSnapshot {
  qpos: Float64Array;
  qvel: Float64Array;
  ctrl: Float64Array;
  warmstart: Float64Array; // MuJoCo's solver warm-start: part of the state for exact replay
  prevAction: Float64Array;
  time: number;
}

export class Sim {
  readonly model: MjModel;
  readonly data: MjData;
  readonly dt: number; // seconds per control step
  mode: Mode = 'ai';
  powered = true;      // false after a knockdown: the robot "powers down" and collapses
  knockdown = '';
  value = 0;           // critic V(s) at the last control step (shown even in Stiff/Limp)
  time = 0;
  lastHits: Hit[] = []; // hits applied in the last control step (effects, debug arrows)
  /** What the network sees (last control step). */
  readonly obs = new Float64Array(OBS_DIM);
  /** What the network outputs, in every mode; only applied to the motors in AI mode (debug panel, ghost). */
  readonly aiAction: Float64Array;

  private prevAction: Float64Array;
  private homeXY: [number, number] = [0, 0];
  private pending: Hit[] = [];
  private falls: FallDetector;
  private actuationBit: number;

  constructor(private mj: MainModule, xml: string, readonly policy: Policy) {
    this.model = mj.MjModel.from_xml_string(xml);
    this.data = new mj.MjData(this.model);
    const c = policy.contract;
    this.dt = c.timestep * c.frame_skip;
    if (Math.abs(this.model.opt.timestep - c.timestep) > 1e-12) {
      throw new Error(`XML timestep ${this.model.opt.timestep} != policy timestep ${c.timestep}`);
    }
    this.prevAction = new Float64Array(c.act_dim);
    this.aiAction = new Float64Array(c.act_dim);
    this.falls = new FallDetector(this.model, c.falls, c.standing_height);
    this.actuationBit = mj.mjtDisableBit.mjDSBL_ACTUATION.value;
    this.reset();
  }

  reset(): void {
    const { model, data } = this;
    this.mj.mj_resetDataKeyframe(model, data, model.key('stand').id);
    data.ctrl.set(this.policy.contract.default_joint_pos);
    this.mj.mj_forward(model, data);
    this.homeXY = [data.qpos[0], data.qpos[1]];
    this.prevAction.fill(0);
    this.pending = [];
    this.lastHits = [];
    this.knockdown = '';
    this.time = 0;
    this.powered = true;
    this.setMode(this.mode);
  }

  setMode(mode: Mode): void {
    this.mode = mode;
    if (mode !== 'ai') this.prevAction.fill(0); // Stiff/Limp hold action 0 = the default pose
    this.applyActuation();
  }

  /** Motors off without changing the selected mode (used after a knockdown). */
  setPowered(on: boolean): void {
    this.powered = on;
    this.applyActuation();
  }

  private applyActuation(): void {
    const opt = this.model.opt;
    const off = this.mode === 'limp' || !this.powered; // same switch as set_limp in Python
    opt.disableflags = off ? opt.disableflags | this.actuationBit : opt.disableflags & ~this.actuationBit;
  }

  /** Queue a hit for the next control step (bullets land on the 50 Hz grid, like training). */
  hit(h: Hit): void {
    this.pending.push(h);
  }

  /** One 50 Hz step. `onSubstep` runs after every 2 ms physics step (replay recording). */
  controlStep(onSubstep?: () => void): void {
    const { model, data, policy } = this;
    const c = policy.contract;
    buildObs(data.qpos, data.qvel, this.prevAction, c.default_joint_pos, this.homeXY, c.home_yaw, c.obs_scales, c.obs_clip, this.obs);
    this.value = policy.value(this.obs);

    // The network always runs (so the debug panel can show what it *would* do in Stiff/Limp),
    // but its output only drives the motors in AI mode.
    this.aiAction.set(policy.act(this.obs));
    const ctrl = data.ctrl;
    if (this.mode === 'ai') {
      for (let i = 0; i < c.act_dim; i++) ctrl[i] = c.default_joint_pos[i] + c.action_scale * this.aiAction[i];
      this.prevAction.set(this.aiAction);
    } else {
      ctrl.set(c.default_joint_pos);
    }

    const hits = this.pending;
    this.pending = [];
    this.lastHits = hits;
    if (hits.length) applyHits(model, data, hits, c.frame_skip);
    for (let i = 0; i < c.frame_skip; i++) {
      this.mj.mj_step(model, data);
      onSubstep?.();
    }
    if (hits.length) clearHits(data);
    this.time += this.dt;

    if (!this.knockdown) this.knockdown = this.falls.check(data);
  }

  snapshot(): SimSnapshot {
    const d = this.data;
    return {
      qpos: Float64Array.from(d.qpos),
      qvel: Float64Array.from(d.qvel),
      ctrl: Float64Array.from(d.ctrl),
      warmstart: Float64Array.from(d.qacc_warmstart),
      prevAction: Float64Array.from(this.prevAction),
      time: this.time,
    };
  }

  restore(s: SimSnapshot): void {
    const d = this.data;
    d.qpos.set(s.qpos);
    d.qvel.set(s.qvel);
    d.ctrl.set(s.ctrl);
    d.xfrc_applied.fill(0);
    this.prevAction.set(s.prevAction);
    this.time = s.time;
    this.pending = [];
    this.lastHits = [];
    this.knockdown = '';
    this.powered = true;
    this.applyActuation();
    this.mj.mj_forward(this.model, d);
    d.qacc_warmstart.set(s.warmstart); // after mj_forward, which overwrites the warm-start
  }
}
