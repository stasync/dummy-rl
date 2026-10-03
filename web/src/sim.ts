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

export class Sim {
  readonly model: MjModel;
  readonly data: MjData;
  readonly dt: number; // seconds per control step
  mode: Mode = 'ai';
  knockdown = '';
  value = 0; // critic V(s) at the last control step (shown even in Stiff/Limp)
  time = 0;

  private obs = new Float64Array(OBS_DIM);
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
    this.knockdown = '';
    this.time = 0;
    this.setMode(this.mode);
  }

  setMode(mode: Mode): void {
    this.mode = mode;
    const opt = this.model.opt;
    // Limp = motors off entirely (same as set_limp in Python).
    opt.disableflags = mode === 'limp' ? opt.disableflags | this.actuationBit : opt.disableflags & ~this.actuationBit;
    if (mode !== 'ai') this.prevAction.fill(0); // Stiff/Limp hold action 0 = the default pose
  }

  /** Queue a hit for the next control step (bullets land on the 50 Hz grid, like training). */
  hit(h: Hit): void {
    this.pending.push(h);
  }

  controlStep(): void {
    const { model, data, policy } = this;
    const c = policy.contract;
    buildObs(data.qpos, data.qvel, this.prevAction, c.default_joint_pos, this.homeXY, c.home_yaw, c.obs_scales, c.obs_clip, this.obs);
    this.value = policy.value(this.obs);

    const ctrl = data.ctrl;
    if (this.mode === 'ai') {
      const a = policy.act(this.obs);
      for (let i = 0; i < a.length; i++) ctrl[i] = c.default_joint_pos[i] + c.action_scale * a[i];
      this.prevAction.set(a);
    } else {
      ctrl.set(c.default_joint_pos);
    }

    const hits = this.pending;
    this.pending = [];
    if (hits.length) applyHits(model, data, hits, c.frame_skip);
    for (let i = 0; i < c.frame_skip; i++) this.mj.mj_step(model, data);
    if (hits.length) clearHits(data);
    this.time += this.dt;

    if (!this.knockdown) this.knockdown = this.falls.check(data);
  }
}
