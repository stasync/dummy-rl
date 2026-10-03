// The trained policy in plain TypeScript: a small MLP forward pass, no ML runtime needed.
// Loads web/public/policy.json written by scripts/export_policy.py (the training <-> game contract).

import type { FallsConfig } from './falls';
import type { ObsScales } from './obs';

export const SCHEMA = 1;

type Activation = 'elu' | 'relu' | 'tanh' | 'none';

interface LayerJson {
  in: number;
  out: number;
  W: number[];
  b: number[];
  act: Activation;
}

export interface Layer {
  in: number;
  out: number;
  W: Float32Array; // row-major out x in; Float32Array = bit-exact copy of the torch weights
  b: Float32Array;
  act: Activation;
}

export interface PolicyContract {
  schema: number;
  meta: { run: string; checkpoint: string; timesteps: number; curriculum_level: number; mujoco: string; xml_sha256: string; exported: string };
  obs_dim: number;
  act_dim: number;
  actor: Layer[];
  critic: Layer[];
  value_range: [number, number];
  obs_scales: ObsScales;
  obs_clip: number;
  default_joint_pos: number[];
  action_scale: number;
  clip_actions: number;
  timestep: number;
  frame_skip: number;
  home_yaw: number;
  standing_height: number;
  falls: FallsConfig;
  hit_model: { j_max_trained: number; j_max_curriculum: number };
}

export function parsePolicy(json: any): PolicyContract {
  if (json.schema !== SCHEMA) {
    throw new Error(`policy.json schema ${json.schema}, game expects ${SCHEMA}: re-export or update policy.ts`);
  }
  const layers = (ls: LayerJson[]): Layer[] =>
    ls.map((l) => ({ in: l.in, out: l.out, act: l.act, W: Float32Array.from(l.W), b: Float32Array.from(l.b) }));
  return { ...json, actor: layers(json.actor), critic: layers(json.critic) };
}

function activate(x: number, act: Activation): number {
  switch (act) {
    case 'elu':
      return x > 0 ? x : Math.expm1(x);
    case 'relu':
      return x > 0 ? x : 0;
    case 'tanh':
      return Math.tanh(x);
    default:
      return x;
  }
}

/** Plain MLP forward pass with preallocated buffers (runs 50x per second, so no garbage). */
class Mlp {
  private bufs: Float64Array[];

  constructor(private layers: Layer[]) {
    this.bufs = layers.map((l) => new Float64Array(l.out));
  }

  forward(x: ArrayLike<number>): Float64Array {
    let input = x;
    for (let li = 0; li < this.layers.length; li++) {
      const { in: n, out: m, W, b, act } = this.layers[li];
      const y = this.bufs[li];
      for (let i = 0; i < m; i++) {
        let s = b[i];
        const row = i * n;
        for (let j = 0; j < n; j++) s += W[row + j] * input[j];
        y[i] = activate(s, act);
      }
      input = y;
    }
    return input as Float64Array;
  }
}

export class Policy {
  readonly contract: PolicyContract;
  private actor: Mlp;
  private critic: Mlp;
  private action: Float64Array;

  constructor(contract: PolicyContract) {
    this.contract = contract;
    this.actor = new Mlp(contract.actor);
    this.critic = new Mlp(contract.critic);
    this.action = new Float64Array(contract.act_dim);
  }

  /** Deterministic action (the Gaussian's mean), clipped like the env does. Reused buffer. */
  act(obs: ArrayLike<number>): Float64Array {
    const mean = this.actor.forward(obs);
    const c = this.contract.clip_actions;
    for (let i = 0; i < this.action.length; i++) this.action[i] = Math.min(c, Math.max(-c, mean[i]));
    return this.action;
  }

  /** Critic's V(s): expected discounted future reward. Drops when the robot is about to fall. */
  value(obs: ArrayLike<number>): number {
    return this.critic.forward(obs)[0];
  }

  /** V(s) mapped to 0..1 with the range measured at export time (the HUD balance meter). */
  confidence(value: number): number {
    const [lo, hi] = this.contract.value_range;
    return Math.min(1, Math.max(0, (value - lo) / (hi - lo)));
  }
}
