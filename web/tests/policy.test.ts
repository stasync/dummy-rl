// The debug panel's attribution (gradient x input via hand-written backprop) must match finite differences.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parsePolicy, Policy } from '../src/policy';

const policy = new Policy(parsePolicy(JSON.parse(readFileSync(resolve(__dirname, '../public/policy.json'), 'utf8'))));
const cases: { obs: number[] }[] = JSON.parse(readFileSync(resolve(__dirname, 'fixtures.json'), 'utf8')).cases;

/** Raw actor mean (no clipping), so finite differences see the true network function. */
function actorMean(x: ArrayLike<number>): Float64Array {
  policy.act(x);
  return Float64Array.from(policy.actorActivations().at(-1)!);
}

describe('actor attribution (debug panel)', () => {
  it('matches finite differences', () => {
    for (const k of cases.slice(0, 5)) {
      const x = Float64Array.from(k.obs);
      policy.act(x);
      const got = policy.attribution(x, new Float64Array(x.length));
      const eps = 1e-6;
      let maxErr = 0, maxVal = 0;
      for (let j = 0; j < x.length; j++) {
        const xp = Float64Array.from(x), xm = Float64Array.from(x);
        xp[j] += eps;
        xm[j] -= eps;
        const fp = actorMean(xp), fm = actorMean(xm);
        let fd = 0;
        for (let o = 0; o < fp.length; o++) fd += Math.abs(((fp[o] - fm[o]) / (2 * eps)) * x[j]);
        maxErr = Math.max(maxErr, Math.abs(fd - got[j]));
        maxVal = Math.max(maxVal, fd);
      }
      expect(maxErr / Math.max(1e-9, maxVal)).toBeLessThan(1e-5);
    }
  });
});
