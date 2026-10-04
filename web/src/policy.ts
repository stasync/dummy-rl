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

/** d act(z) / dz written in terms of the activation's *output* y (so pre-activations needn't be stored). */
function activationSlope(y: number, act: Activation): number {
  switch (act) {
    case 'elu':
      return y > 0 ? 1 : y + 1; // y = e^z - 1  ->  dy/dz = e^z = y + 1
    case 'relu':
      return y > 0 ? 1 : 0;
    case 'tanh':
      return 1 - y * y;
    default:
      return 1;
  }
}

/** Plain MLP forward pass with preallocated buffers (runs 50x per second, so no garbage). */
class Mlp {
  private bufs: Float64Array[];
  private grads: Float64Array[]; // backprop scratch, one per layer input
  private seed: Float64Array;

  constructor(private layers: Layer[]) {
    this.bufs = layers.map((l) => new Float64Array(l.out));
    this.grads = layers.map((l) => new Float64Array(l.in));
    this.seed = new Float64Array(layers[layers.length - 1].out);
  }

  /** Each layer's output (after its activation) from the last forward() call. */
  activations(): Float64Array[] {
    return this.bufs;
  }

  /**
   * Gradient x input attribution: out[j] = sum_k |d y_k / d x_j * x_j| over all outputs k.
   * "How much input j is moving the outputs right now". Inputs near zero (e.g. velocities while
   * standing still) contribute nothing; after a hit, the ones driving the reaction light up.
   * Uses the activations of the last forward(x) call, so call it right after forward with the same x.
   */
  attribution(x: ArrayLike<number>, out: Float64Array): Float64Array {
    out.fill(0);
    const L = this.layers.length;
    const last = this.layers[L - 1];
    for (let k = 0; k < last.out; k++) {
      // Backprop output k: start with d y_k / d z_last = e_k * act'(.)
      this.seed.fill(0);
      this.seed[k] = activationSlope(this.bufs[L - 1][k], last.act);
      let dz: Float64Array = this.seed;
      for (let li = L - 1; li >= 0; li--) {
        const { in: n, out: m, W } = this.layers[li];
        const dy = this.grads[li]; // d y_k / d (input of layer li) = W^T dz
        dy.fill(0);
        for (let i = 0; i < m; i++) {
          const gi = dz[i];
          if (gi === 0) continue;
          const row = i * n;
          for (let j = 0; j < n; j++) dy[j] += W[row + j] * gi;
        }
        if (li > 0) {
          const prev = this.bufs[li - 1], act = this.layers[li - 1].act;
          for (let j = 0; j < n; j++) dy[j] *= activationSlope(prev[j], act); // through the previous activation
        }
        dz = dy;
      }
      for (let j = 0; j < out.length; j++) out[j] += Math.abs(dz[j] * x[j]);
    }
    return out;
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

  /** Hidden + output activations of the actor from the last act() call (debug panel). */
  actorActivations(): Float64Array[] {
    return this.actor.activations();
  }

  /** Gradient x input attribution of the actor for `obs` (call right after act(obs)). */
  attribution(obs: ArrayLike<number>, out: Float64Array): Float64Array {
    return this.actor.attribution(obs, out);
  }

  /** V(s) mapped to 0..1 with the range measured at export time (the HUD balance meter). */
  confidence(value: number): number {
    const [lo, hi] = this.contract.value_range;
    return Math.min(1, Math.max(0, (value - lo) / (hi - lo)));
  }
}
