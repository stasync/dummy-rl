// Tiny synthesized sounds (WebAudio, no audio files): shots, metal impacts, knockdown thud.
// The AudioContext can only start after a user gesture, so it is created on the first shot.

import type { WeaponId } from './weapons';

export class Sfx {
  private ctx: AudioContext | null = null;
  private noise: AudioBuffer | null = null;
  private master: GainNode | null = null;

  private ensure(): AudioContext | null {
    if (this.ctx) return this.ctx;
    try {
      this.ctx = new AudioContext();
    } catch {
      return null; // no audio available: the game still works silently
    }
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.5;
    this.master.connect(this.ctx.destination);
    const len = this.ctx.sampleRate;
    this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const ch = this.noise.getChannelData(0);
    for (let i = 0; i < len; i++) ch[i] = Math.random() * 2 - 1;
    return this.ctx;
  }

  private burst(freq: number, q: number, gain: number, decay: number, delay = 0): void {
    const ctx = this.ensure();
    if (!ctx || !this.noise || !this.master) return;
    const t = ctx.currentTime + delay;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const f = ctx.createBiquadFilter();
    f.type = 'bandpass';
    f.frequency.value = freq;
    f.Q.value = q;
    const g = ctx.createGain();
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + decay);
    src.connect(f).connect(g).connect(this.master);
    src.start(t, Math.random() * 0.5, decay + 0.05);
  }

  private tone(type: OscillatorType, f0: number, f1: number, gain: number, decay: number, delay = 0): void {
    const ctx = this.ensure();
    if (!ctx || !this.master) return;
    const t = ctx.currentTime + delay;
    const o = ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f0, t);
    o.frequency.exponentialRampToValueAtTime(f1, t + decay);
    const g = ctx.createGain();
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + decay);
    o.connect(g).connect(this.master);
    o.start(t);
    o.stop(t + decay + 0.05);
  }

  shot(weapon: WeaponId): void {
    switch (weapon) {
      case 'pistol':
        this.burst(1800, 0.7, 0.9, 0.12);
        this.tone('sine', 160, 50, 0.5, 0.1);
        break;
      case 'rifle':
        this.burst(2400, 0.9, 0.6, 0.07);
        this.tone('sine', 200, 70, 0.3, 0.06);
        break;
      case 'shotgun':
        this.burst(900, 0.5, 1.0, 0.25);
        this.tone('sine', 110, 40, 0.8, 0.2);
        break;
      case 'cannon':
        this.burst(300, 0.4, 1.0, 0.6);
        this.tone('sine', 70, 25, 1.0, 0.7);
        break;
    }
  }

  /** Metal clank; strength in 0..1+. */
  impact(strength: number): void {
    const s = Math.min(1.5, strength);
    const base = 700 + Math.random() * 500;
    this.tone('triangle', base, base * 0.8, 0.25 * s, 0.18, 0.01);
    this.tone('square', base * 1.51, base * 1.3, 0.06 * s, 0.12, 0.01);
    this.burst(4000, 1.5, 0.3 * s, 0.05, 0.01);
  }

  knockdown(): void {
    this.tone('sine', 90, 30, 0.9, 0.5);
    this.burst(500, 0.6, 0.6, 0.35, 0.05);
    this.impact(0.6);
  }
}
