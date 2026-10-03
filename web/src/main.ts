// Entry point (Day 2 dev build): robot in the browser, live policy, Stiff/Limp/AI modes,
// click-to-shove as a debug hit. Weapons, score, replay etc. come on Day 3.

import * as THREE from 'three';
import loadMujoco from '@mujoco/mujoco';
import xml from '../../assets/stagger.xml?raw';
import { parsePolicy, Policy } from './policy';
import { Scene } from './scene';
import { Sim, type Mode } from './sim';

const DEBUG_SHOVE_FRACTION = 0.75; // debug shove = 75% of the strongest hit the policy trained on
const MAX_FRAME_S = 0.1;           // clamp long frames (tab switch) instead of fast-forwarding

const $ = (id: string) => document.getElementById(id)!;

async function sha256(text: string): Promise<string> {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function main(): Promise<void> {
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
  const view = new Scene($('view') as unknown as HTMLCanvasElement, mj, sim.model);
  const shoveJ = DEBUG_SHOVE_FRACTION * c.hit_model.j_max_trained;
  $('help').textContent = `click: shove ${shoveJ.toFixed(0)} N·s · 1 AI · 2 Stiff · 3 Limp · R reset`;
  $('dev').textContent = `dev build · policy ${c.meta.run} (${(c.meta.timesteps / 1e6).toFixed(1)}M steps, level ${c.meta.curriculum_level})`;

  const modeNames: Record<Mode, string> = { ai: '1 AI', stiff: '2 Stiff', limp: '3 Limp' };
  const drawModes = () => {
    $('modes').innerHTML = (Object.keys(modeNames) as Mode[])
      .map((m) => `<span class="${m === sim.mode ? 'on' : ''}">${modeNames[m]}</span>`)
      .join('');
  };
  drawModes();

  window.addEventListener('keydown', (e) => {
    const modes: Record<string, Mode> = { '1': 'ai', '2': 'stiff', '3': 'limp' };
    if (modes[e.key]) {
      sim.setMode(modes[e.key]);
      drawModes();
    } else if (e.key === 'r' || e.key === 'R') {
      sim.reset();
    }
  });

  // Debug shove: raycast from the camera through the cursor; push along the ray at the hit point.
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  view.renderer.domElement.addEventListener('pointerdown', (e) => {
    ndc.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
    raycaster.setFromCamera(ndc, view.camera);
    const hit = raycaster.intersectObjects(view.robotMeshes, false)[0];
    if (!hit) return;
    const d = raycaster.ray.direction;
    sim.hit({
      bodyId: hit.object.userData.bodyId,
      point: [hit.point.x, hit.point.y, hit.point.z],
      impulse: [d.x * shoveJ, d.y * shoveJ, d.z * shoveJ],
    });
  });

  // Fixed-timestep loop: physics/policy advance in exact 20 ms control steps regardless of
  // display refresh rate, so the policy sees the same timing it was trained with.
  let last = performance.now();
  let acc = 0;
  const frame = (now: number) => {
    acc += Math.min(MAX_FRAME_S, (now - last) / 1000);
    last = now;
    while (acc >= sim.dt) {
      sim.controlStep();
      acc -= sim.dt;
    }
    view.update(sim.data);
    view.render();

    const status = $('status');
    status.textContent = sim.knockdown ? `KNOCKDOWN (${sim.knockdown}) · R to reset` : `standing · ${sim.time.toFixed(1)} s`;
    status.className = sim.knockdown ? 'ko' : '';
    $('meter-fill').style.width = `${(policy.confidence(sim.value) * 100).toFixed(0)}%`;
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  $('status').textContent = `failed to start: ${err.message ?? err}`;
});
