// WASM smoke test (PLAN.md §4.1 acceptance): the exact XML used for training loads in
// @mujoco/mujoco under Node, Stiff mode stands, Limp mode collapses.
// Also checks the WASM sim tracks the native Python sim (early warning before the full
// parity test in PLAN.md §7).
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import loadMujoco, { type MainModule } from '@mujoco/mujoco';

const XML_PATH = resolve(__dirname, '../../assets/stagger.xml');
const REF_PATH = resolve(__dirname, 'fixtures/smoke_ref.json');

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

function load() {
  const model = mj.MjModel.from_xml_string(readFileSync(XML_PATH, 'utf8'));
  const data = new mj.MjData(model);
  mj.mj_resetDataKeyframe(model, data, 0);
  return { model, data };
}

// Torso "up" component: world z of the torso's z axis (1 = upright, 0 = lying flat).
function torsoUp(model: any, data: any): number {
  const id = model.body('torso').id;
  return data.xmat[id * 9 + 8];
}

describe('stagger.xml in MuJoCo WASM', () => {
  it('loads with the expected sizes and mass', () => {
    const { model, data } = load();
    expect(model.nq).toBe(25);
    expect(model.nv).toBe(24);
    expect(model.nu).toBe(18);
    expect(model.body_subtreemass[1]).toBeCloseTo(30.42, 2);
    data.delete();
    model.delete();
  });

  it('Stiff mode (hold default pose) stands for 10 s', () => {
    const { model, data } = load();
    for (let i = 0; i < 5000; i++) mj.mj_step(model, data);
    expect(torsoUp(model, data)).toBeGreaterThan(0.99);
    expect(data.qpos[2]).toBeGreaterThan(0.68);
    data.delete();
    model.delete();
  });

  it('Limp mode (actuation disabled) collapses', () => {
    const { model, data } = load();
    model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_ACTUATION.value;
    for (let i = 0; i < 1500; i++) mj.mj_step(model, data);
    expect(data.qpos[2]).toBeLessThan(0.4);
    data.delete();
    model.delete();
  });

  it('tracks the native Python sim (1 s of Stiff, from fixtures/smoke_ref.json)', () => {
    const ref = JSON.parse(readFileSync(REF_PATH, 'utf8'));
    const { model, data } = load();
    for (let i = 0; i < ref.steps; i++) mj.mj_step(model, data);
    let maxDiff = 0;
    for (let i = 0; i < model.nq; i++) maxDiff = Math.max(maxDiff, Math.abs(data.qpos[i] - ref.qpos[i]));
    console.log(`max |qpos_wasm - qpos_python| after ${ref.steps} steps: ${maxDiff.toExponential(2)}`);
    expect(maxDiff).toBeLessThan(1e-6);
    data.delete();
    model.delete();
  });
});
