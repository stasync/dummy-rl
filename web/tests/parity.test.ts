// Python <-> TypeScript parity (PLAN.md §7). Run after touching obs, hits, falls or the export.
//
//   python scripts/export_policy.py   --run runs/<run> --out web/public/policy.json
//   python scripts/parity_fixtures.py --run runs/<run> --out web/tests/fixtures.json
//   cd web && npx vitest run
//
// Each fixture is a state the robot actually visited while being shot at. For each one:
//   obs, action, value   must match Python (obs/action abs diff < 1e-4, value relative < 1e-5)
//   knockdown            must give the same reason string
//   applyHit + one control step must give the same qvel (< 1e-6) and knockdown reason
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import loadMujoco, { type MainModule, type MjModel } from '@mujoco/mujoco';
import { buildObs } from '../src/obs';
import { applyHits, clearHits, type Vec3 } from '../src/hits';
import { FallDetector } from '../src/falls';
import { Policy, parsePolicy } from '../src/policy';

const XML_PATH = resolve(__dirname, '../../assets/stagger.xml');
const POLICY_PATH = resolve(__dirname, '../public/policy.json');
const FIXTURES_PATH = resolve(__dirname, 'fixtures.json');

interface Case {
  qpos: number[];
  qvel: number[];
  prev_action: number[];
  home_xy: number[];
  home_yaw: number;
  obs: number[];
  action: number[];
  value: number;
  ctrl: number[];
  knockdown: string;
  hit: { body_id: number; point: Vec3; impulse: Vec3 };
  qvel_after_hit: number[];
  knockdown_after_hit: string;
}

const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

let mj: MainModule;
let model: MjModel;
let policy: Policy;
let falls: FallDetector;
const xml = readFileSync(XML_PATH, 'utf8');
const fixtures: { run: string; mujoco: string; cases: Case[] } = JSON.parse(readFileSync(FIXTURES_PATH, 'utf8'));

beforeAll(async () => {
  mj = await loadMujoco();
  model = mj.MjModel.from_xml_string(xml);
  policy = new Policy(parsePolicy(JSON.parse(readFileSync(POLICY_PATH, 'utf8'))));
  falls = new FallDetector(model, policy.contract.falls, policy.contract.standing_height);
});

describe('policy.json contract', () => {
  it('was exported for this exact robot XML and MuJoCo version', () => {
    const c = policy.contract;
    expect(c.meta.xml_sha256).toBe(createHash('sha256').update(xml).digest('hex'));
    expect(c.meta.mujoco).toBe(fixtures.mujoco);
    expect(c.meta.run).toBe(fixtures.run);
    expect([c.obs_dim, c.act_dim]).toEqual([67, model.nu]);
  });
});

describe(`parity with Python (${fixtures.cases.length} cases)`, () => {
  it('obs, action and value match', () => {
    let obsDiff = 0, actDiff = 0, valRel = 0;
    const c = policy.contract;
    for (const k of fixtures.cases) {
      const obs = buildObs(k.qpos, k.qvel, k.prev_action, c.default_joint_pos, k.home_xy, k.home_yaw, c.obs_scales, c.obs_clip);
      obsDiff = Math.max(obsDiff, maxAbsDiff(obs, k.obs));
      actDiff = Math.max(actDiff, maxAbsDiff(policy.act(obs), k.action));
      valRel = Math.max(valRel, Math.abs(policy.value(obs) - k.value) / Math.max(1, Math.abs(k.value)));
    }
    console.log(`obs ${obsDiff.toExponential(2)}  action ${actDiff.toExponential(2)}  value(rel) ${valRel.toExponential(2)}`);
    expect(obsDiff).toBeLessThan(1e-4);
    expect(actDiff).toBeLessThan(1e-4);
    expect(valRel).toBeLessThan(1e-5);
  });

  it('knockdown detection and applyHit + one control step match', () => {
    const frameSkip = policy.contract.frame_skip;
    let qvelDiff = 0;
    let knockdowns = 0;
    for (const [i, k] of fixtures.cases.entries()) {
      const data = new mj.MjData(model); // fresh: MuJoCo warm-starts its solver from the last step
      try {
        data.qpos.set(k.qpos);
        data.qvel.set(k.qvel);
        data.ctrl.set(k.ctrl);
        mj.mj_forward(model, data);
        expect(falls.check(data), `case ${i} before hit`).toBe(k.knockdown);
        knockdowns += k.knockdown ? 1 : 0;

        applyHits(model, data, [{ bodyId: k.hit.body_id, point: k.hit.point, impulse: k.hit.impulse }], frameSkip);
        for (let s = 0; s < frameSkip; s++) mj.mj_step(model, data);
        clearHits(data);
        qvelDiff = Math.max(qvelDiff, maxAbsDiff(data.qvel, k.qvel_after_hit));
        expect(falls.check(data), `case ${i} after hit`).toBe(k.knockdown_after_hit);
      } finally {
        data.delete();
      }
    }
    console.log(`qvel after hit ${qvelDiff.toExponential(2)}  (${knockdowns} knocked-down cases)`);
    expect(knockdowns).toBeGreaterThan(0); // the fixtures must actually exercise the knockdown path
    expect(qvelDiff).toBeLessThan(1e-6);
  });
});
