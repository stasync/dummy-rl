// The game's Sim class, headless: the trained robot reacts to a debug shove (Day 2 "done when").
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { parsePolicy, Policy } from '../src/policy';
import { Sim, type Mode } from '../src/sim';

const xml = readFileSync(resolve(__dirname, '../../assets/stagger.xml'), 'utf8');
let mj: MainModule;
let policy: Policy;

beforeAll(async () => {
  mj = await loadMujoco();
  policy = new Policy(parsePolicy(JSON.parse(readFileSync(resolve(__dirname, '../public/policy.json'), 'utf8'))));
});

/** Settle 1 s, shove the chest from the shooter's side (+x, pushing toward -x), watch 3 s. */
function shove(mode: Mode, j: number) {
  const sim = new Sim(mj, xml, policy);
  sim.setMode(mode);
  for (let i = 0; i < 50; i++) sim.controlStep();
  const torso = sim.model.body('torso').id;
  const p = sim.data.xipos;
  sim.hit({ bodyId: torso, point: [p[torso * 3] + 0.09, p[torso * 3 + 1], p[torso * 3 + 2] + 0.05], impulse: [-j, 0, 0] });
  let minConfidence = 1;
  for (let i = 0; i < 150; i++) {
    sim.controlStep();
    minConfidence = Math.min(minConfidence, policy.confidence(sim.value));
  }
  const result = { knockdown: sim.knockdown, drift: Math.hypot(sim.data.qpos[0], sim.data.qpos[1]), minConfidence };
  sim.data.delete();
  sim.model.delete();
  return result;
}

describe('game Sim (headless)', () => {
  it('stands untouched in every mode except Limp', () => {
    for (const mode of ['ai', 'stiff', 'limp'] as Mode[]) {
      const sim = new Sim(mj, xml, policy);
      sim.setMode(mode);
      for (let i = 0; i < 250; i++) sim.controlStep(); // 5 s
      expect(sim.knockdown === '', `${mode}: ${sim.knockdown}`).toBe(mode !== 'limp');
      sim.data.delete();
      sim.model.delete();
    }
  });

  it('AI recovers from a debug shove that topples Stiff', () => {
    const j = 0.75 * policy.contract.hit_model.j_max_trained;
    const ai = shove('ai', j);
    const stiff = shove('stiff', j);
    console.log(`shove ${j} N·s  AI: ${JSON.stringify(ai)}  Stiff: ${JSON.stringify(stiff)}`);
    expect(ai.knockdown).toBe('');
    expect(ai.minConfidence).toBeLessThan(0.9); // the balance meter reacts to the shove
  });
});
