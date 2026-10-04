// The game loop and rules: shoot -> robot reacts -> knockdown -> slow-mo replay -> rewind or new round.
//
// Score = shots fired until the knockdown (fewer is better), per weapon. A round only sets a
// record if it was played in AI mode with a single weapon (Stiff/Limp are too easy).

import * as THREE from 'three';
import type { MainModule, MjData } from '@mujoco/mujoco';
import type { Sfx } from './audio';
import type { DebugOverlay } from './debug';
import type { Hud } from './hud';
import type { Scene } from './scene';
import type { Mode, Sim, SimSnapshot } from './sim';
import { makeWeapons, type Weapon, type WeaponId } from './weapons';

type Phase = 'playing' | 'down' | 'replay' | 'review';

const REPLAY_BEFORE_S = 3.0;     // replay starts this long before the knockdown...
const REPLAY_AFTER_S = 1.2;      // ...and ends this long after it
const REPLAY_SPEED = 0.3;        // slow motion factor
const POWER_DOWN_DELAY_S = 0.25; // after a KO the controller gives up and the motors switch off
const HIT_STOP_S = 0.06;         // freeze physics briefly on big hits (sells the impact)
const BIG_HIT = 0.9;             // "big" = at least this fraction of j_max_trained in one shot
const MAX_FRAME_S = 0.1;         // clamp long frames (tab switch) instead of fast-forwarding
const BEST_KEY = 'stagger.best.v1';

const KO_TEXT: Record<string, string> = { height: 'it folded', tilt: 'it toppled' };

/** Ring buffer of qpos, one entry per 2 ms physics step: the slow-mo replay plays these back. */
class ReplayBuffer {
  total = 0; // frames ever pushed (stable index across wrap-around)
  private buf: Float64Array;

  constructor(private nq: number, private capacity: number) {
    this.buf = new Float64Array(nq * capacity);
  }

  push(qpos: ArrayLike<number>): void {
    this.buf.set(qpos, (this.total % this.capacity) * this.nq);
    this.total++;
  }

  /** Frame by its stable index, or null if it was overwritten / not recorded yet. */
  frame(index: number): Float64Array | null {
    if (index < 0 || index >= this.total || index < this.total - this.capacity) return null;
    const o = (index % this.capacity) * this.nq;
    return this.buf.subarray(o, o + this.nq);
  }

  clear(): void {
    this.total = 0;
  }
}

interface ShotRecord {
  snap: SimSnapshot;
  shots: number;
  weaponsUsed: Set<WeaponId>;
  aiOnly: boolean;
}

export class Game {
  private phase: Phase = 'playing';
  private phaseTime = 0;
  private weapons: Weapon[];
  private weaponIdx = 0;
  private shots = 0;
  private weaponsUsed = new Set<WeaponId>();
  private aiOnly = true;
  private lastShot: ShotRecord | null = null;
  private cooldown = 0;
  private triggerHeld = false;
  private pointer = new THREE.Vector2();
  private raycaster = new THREE.Raycaster();
  private acc = 0;
  private hitStop = 0;
  private replay: ReplayBuffer;
  private replayData: MjData;
  private replayFrom = 0;
  private replayTo = 0;
  private replayT = 0;
  private best: Partial<Record<WeaponId, number>> = loadBest();
  private jTrained: number;
  private physDt: number;

  constructor(
    private mj: MainModule,
    private sim: Sim,
    private scene: Scene,
    private hud: Hud,
    private sfx: Sfx,
    private debug: DebugOverlay,
  ) {
    const c = sim.policy.contract;
    this.jTrained = c.hit_model.j_max_trained;
    this.weapons = makeWeapons(this.jTrained);
    this.physDt = c.timestep;
    const seconds = REPLAY_BEFORE_S + REPLAY_AFTER_S + 0.5;
    this.replay = new ReplayBuffer(sim.model.nq, Math.ceil(seconds / this.physDt));
    this.replayData = new mj.MjData(sim.model);
    this.bindInput();
    hud.setMode(sim.mode);
    this.refreshHud();
  }

  private get weapon(): Weapon {
    return this.weapons[this.weaponIdx];
  }

  // --- input --------------------------------------------------------------------------------

  private bindInput(): void {
    const canvas = this.scene.renderer.domElement;
    canvas.addEventListener('pointermove', (e) => this.setPointer(e));
    canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      this.setPointer(e);
      if (this.phase === 'replay') return this.enterReview();
      this.triggerHeld = true;
      this.fire();
    });
    window.addEventListener('pointerup', () => (this.triggerHeld = false));
    canvas.addEventListener('pointerleave', () => (this.triggerHeld = false));
    canvas.addEventListener('wheel', (e) => this.cycleWeapon(e.deltaY > 0 ? 1 : -1), { passive: true });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    window.addEventListener('keydown', (e) => {
      const modes: Record<string, Mode> = { '1': 'ai', '2': 'stiff', '3': 'limp' };
      if (modes[e.key]) return this.setMode(modes[e.key]);
      switch (e.key.toLowerCase()) {
        case 'q': return this.cycleWeapon(-1);
        case 'e': return this.cycleWeapon(1);
        case 'r': return this.rewind();
        case 'enter': return this.newRound();
        case ' ':
          e.preventDefault();
          if (this.phase === 'replay') return this.enterReview();
          if (this.phase === 'review') return this.newRound();
          return;
      }
      if (e.code === 'Backquote') this.debug.toggle();
    });
  }

  private setPointer(e: PointerEvent): void {
    this.pointer.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  }

  private setMode(mode: Mode): void {
    this.sim.setMode(mode);
    if (mode !== 'ai') this.aiOnly = false;
    this.hud.setMode(mode);
    this.refreshHud();
  }

  private cycleWeapon(step: number): void {
    this.weaponIdx = (this.weaponIdx + step + this.weapons.length) % this.weapons.length;
    this.cooldown = 0;
    this.refreshHud();
  }

  // --- shooting -----------------------------------------------------------------------------

  private fire(): void {
    if (this.phase !== 'playing' || this.cooldown > 0) return;
    const w = this.weapon;
    this.cooldown = w.cooldown;
    // Remember the state *before* this shot lands: R rewinds to here after a knockdown.
    this.lastShot = { snap: this.sim.snapshot(), shots: this.shots, weaponsUsed: new Set(this.weaponsUsed), aiOnly: this.aiOnly };
    this.shots++;
    this.weaponsUsed.add(w.id);
    this.sfx.shot(w.id);

    this.raycaster.setFromCamera(this.pointer, this.scene.camera);
    const origin = this.raycaster.ray.origin.clone();
    const aim = this.raycaster.ray.direction.clone();
    let delivered = 0;
    for (let i = 0; i < w.pellets; i++) {
      const dir = spread(aim, w.spreadDeg);
      this.raycaster.set(origin, dir);
      const hit = this.raycaster.intersectObjects(this.scene.robotMeshes, false)[0];
      if (hit) {
        const J = dir.clone().multiplyScalar(w.impulse);
        this.sim.hit({ bodyId: hit.object.userData.bodyId, point: [hit.point.x, hit.point.y, hit.point.z], impulse: [J.x, J.y, J.z] });
        this.scene.spark(hit.point, dir.clone().negate(), w.impulse / this.jTrained);
        delivered += w.impulse;
        continue;
      }
      const miss = this.raycaster.intersectObjects(this.scene.rangeMeshes, false)[0];
      if (miss) this.scene.spark(miss.point, miss.face?.normal ?? dir.clone().negate(), 0.15);
    }
    if (delivered > 0) this.sfx.impact(delivered / this.jTrained);
    if (delivered >= BIG_HIT * this.jTrained) this.hitStop = HIT_STOP_S;
    this.scene.shake(Math.min(1, 0.08 + (0.35 * delivered) / this.jTrained));
    this.refreshHud();
  }

  // --- round flow ---------------------------------------------------------------------------

  private onKnockdown(): void {
    this.phase = 'down';
    this.phaseTime = 0;
    this.replayTo = this.replay.total + Math.round(REPLAY_AFTER_S / this.physDt);
    this.replayFrom = Math.max(0, this.replay.total - Math.round(REPLAY_BEFORE_S / this.physDt));
    this.sfx.knockdown();
    this.scene.shake(0.5);

    const w = this.weapon.id;
    const recordable = this.aiOnly && this.weaponsUsed.size === 1;
    if (recordable && (this.best[w] === undefined || this.shots < this.best[w]!)) {
      this.best[w] = this.shots;
      saveBest(this.best);
      this.hud.flashRecord();
    }
    this.hud.banner('KNOCKDOWN', `${this.shots} shot${this.shots === 1 ? '' : 's'} · ${this.koText()}`);
    this.refreshHud();
  }

  private koText(): string {
    const k = this.sim.knockdown;
    return k.startsWith('contact:') ? `${k.slice(8).replace('_', ' ')} hit the floor` : KO_TEXT[k] ?? k;
  }

  private startReplay(): void {
    this.phase = 'replay';
    this.replayT = 0;
    this.hud.banner('REPLAY', `${REPLAY_SPEED}× · click or <kbd>Space</kbd> to skip`);
  }

  private enterReview(): void {
    this.phase = 'review';
    this.hud.banner(
      'KNOCKDOWN',
      `${this.shots} shot${this.shots === 1 ? '' : 's'} · ${this.koText()}<br>` +
        `<kbd>R</kbd> rewind to before the last shot · <kbd>Enter</kbd> new round`,
    );
  }

  private rewind(): void {
    if (this.phase === 'playing' || !this.lastShot) return;
    const s = this.lastShot;
    this.sim.restore(s.snap);
    this.shots = s.shots;
    this.weaponsUsed = new Set(s.weaponsUsed);
    this.aiOnly = s.aiOnly;
    this.resumePlay();
  }

  private newRound(): void {
    this.sim.reset();
    this.shots = 0;
    this.weaponsUsed.clear();
    this.aiOnly = this.sim.mode === 'ai';
    this.lastShot = null;
    this.resumePlay();
  }

  private resumePlay(): void {
    this.phase = 'playing';
    this.acc = 0;
    this.hitStop = 0;
    this.replay.clear();
    this.hud.banner(null);
    this.refreshHud();
  }

  private refreshHud(): void {
    this.hud.setWeapons(this.weapons, this.weaponIdx);
    const note = !this.aiOnly ? 'records only in AI mode' : this.weaponsUsed.size > 1 ? 'mixed weapons: no record' : '';
    this.hud.setScore(this.shots, this.best[this.weapon.id], note);
  }

  // --- per frame ----------------------------------------------------------------------------

  frame(dtReal: number): void {
    const dt = Math.min(MAX_FRAME_S, dtReal);
    this.cooldown = Math.max(0, this.cooldown - dt);
    this.phaseTime += dt;
    if (this.triggerHeld && this.weapon.auto) this.fire();

    let shown: MjData = this.sim.data;
    if (this.phase === 'replay') {
      shown = this.replayFrame(dt);
    } else {
      this.stepPhysics(dt);
      if (this.phase === 'down') {
        if (this.phaseTime > POWER_DOWN_DELAY_S && this.sim.powered) this.sim.setPowered(false);
        if (this.replay.total >= this.replayTo) this.startReplay();
      }
    }

    this.scene.update(shown);
    this.debug.update(shown, dt);
    this.hud.setMeter(this.sim.policy.confidence(this.sim.value));
    if (this.phase === 'playing') {
      this.hud.setStatus(`${this.sim.mode === 'ai' ? 'AI balancing' : this.sim.mode === 'stiff' ? 'holding pose (Stiff)' : 'motors off (Limp)'} · ${this.sim.time.toFixed(1)} s`);
    }
    this.scene.render(dt);
  }

  private stepPhysics(dt: number): void {
    if (this.hitStop > 0) {
      this.hitStop -= dt;
      return;
    }
    // Fixed timestep: physics/policy advance in exact 20 ms control steps regardless of the
    // display's refresh rate, so the policy sees the same timing it was trained with.
    this.acc += dt;
    const record = () => this.replay.push(this.sim.data.qpos);
    while (this.acc >= this.sim.dt) {
      this.sim.controlStep(record);
      this.acc -= this.sim.dt;
      this.debug.addHits(this.sim.lastHits);
      if (this.phase === 'playing' && this.sim.knockdown) this.onKnockdown();
    }
  }

  private replayFrame(dt: number): MjData {
    this.replayT += dt * REPLAY_SPEED;
    const idx = this.replayFrom + Math.floor(this.replayT / this.physDt);
    const frame = this.replay.frame(Math.min(idx, this.replayTo - 1));
    if (idx >= this.replayTo - 1 || !frame) this.enterReview();
    if (!frame) return this.sim.data;
    this.replayData.qpos.set(frame);
    this.mj.mj_kinematics(this.sim.model, this.replayData); // positions only: playback, not simulation
    return this.replayData;
  }
}

/** Random direction inside a cone of half-angle `deg` around `dir` (uniform over the spherical cap). */
function spread(dir: THREE.Vector3, deg: number): THREE.Vector3 {
  if (deg <= 0) return dir.clone();
  const cosMax = Math.cos((deg * Math.PI) / 180);
  const cosA = 1 - Math.random() * (1 - cosMax);
  const sinA = Math.sqrt(1 - cosA * cosA);
  const phi = Math.random() * Math.PI * 2;
  const up = Math.abs(dir.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
  const b1 = new THREE.Vector3().crossVectors(dir, up).normalize();
  const b2 = new THREE.Vector3().crossVectors(dir, b1);
  return dir.clone().multiplyScalar(cosA).addScaledVector(b1, sinA * Math.cos(phi)).addScaledVector(b2, sinA * Math.sin(phi)).normalize();
}

function loadBest(): Partial<Record<WeaponId, number>> {
  try {
    return JSON.parse(localStorage.getItem(BEST_KEY) ?? '{}');
  } catch {
    return {};
  }
}

function saveBest(best: Partial<Record<WeaponId, number>>): void {
  try {
    localStorage.setItem(BEST_KEY, JSON.stringify(best));
  } catch {
    // storage blocked (private mode, itch.io iframe settings): scores just don't persist
  }
}
