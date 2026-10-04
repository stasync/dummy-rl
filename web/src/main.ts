// Entry point: load MuJoCo (WASM), the robot XML and the trained policy, then run the game.

import loadMujoco from '@mujoco/mujoco';
import xml from '../../assets/stagger.xml?raw';
import { Sfx } from './audio';
import { DebugOverlay } from './debug';
import { Game } from './game';
import { Hud } from './hud';
import { parsePolicy, Policy } from './policy';
import { Scene } from './scene';
import { Sim } from './sim';

async function sha256(text: string): Promise<string> {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function main(): Promise<void> {
  const hud = new Hud();
  const [mj, policyJson] = await Promise.all([
    loadMujoco(),
    fetch(`${import.meta.env.BASE_URL}policy.json`).then((r) => r.json()),
  ]);
  const policy = new Policy(parsePolicy(policyJson));
  const c = policy.contract;
  if (crypto.subtle && (await sha256(xml)) !== c.meta.xml_sha256) {
    console.warn('policy.json was trained on a different stagger.xml; re-export the policy');
  }

  const sim = new Sim(mj, xml, policy);
  const pelvis = sim.model.body('pelvis').id;
  const standingComHeight = sim.data.subtree_com[pelvis * 3 + 2]; // reset() ran mj_forward
  const scene = new Scene(document.getElementById('view') as HTMLCanvasElement, mj, sim.model);
  const debug = new DebugOverlay(mj, sim.model, scene, standingComHeight);
  const game = new Game(mj, sim, scene, hud, new Sfx(), debug);
  hud.setInfo(`policy ${c.meta.run} · ${(c.meta.timesteps / 1e6).toFixed(1)}M steps · trained to ${c.hit_model.j_max_trained} N·s`);

  let last = performance.now();
  const loop = (now: number) => {
    game.frame((now - last) / 1000);
    last = now;
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

main().catch((err) => {
  console.error(err);
  document.getElementById('status')!.textContent = `failed to start: ${err.message ?? err}`;
});
