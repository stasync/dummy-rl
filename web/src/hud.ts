// DOM HUD: robot mode, weapons, shot counter + best score, balance meter, banners.

import type { Mode } from './sim';
import type { Weapon } from './weapons';

const $ = (id: string) => document.getElementById(id)!;

const MODE_LABEL: Record<Mode, string> = { ai: 'AI', stiff: 'Stiff', limp: 'Limp' };

export class Hud {
  private crosshair = $('crosshair');

  constructor() {
    window.addEventListener('pointermove', (e) => {
      this.crosshair.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    });
  }

  setInfo(text: string): void {
    $('dev').textContent = text;
  }

  setMode(mode: Mode): void {
    $('modes').innerHTML = (Object.keys(MODE_LABEL) as Mode[])
      .map((m, i) => `<span class="${m === mode ? 'on' : ''}">${i + 1} ${MODE_LABEL[m]}</span>`)
      .join('');
  }

  setWeapons(weapons: Weapon[], current: number): void {
    $('weapons').innerHTML = weapons
      .map((w, i) => `<div class="${i === current ? 'on' : ''}"><b>${w.name}</b> <span>${w.impulse} N·s${w.pellets > 1 ? ` ×${w.pellets}` : ''}</span></div>`)
      .join('') + `<div class="note">${weapons[current].note}</div>`;
  }

  setScore(shots: number, best: number | undefined, recordable: string): void {
    $('shots').textContent = String(shots);
    $('best').textContent = best === undefined ? '–' : String(best);
    $('record-note').textContent = recordable;
  }

  setMeter(confidence: number): void {
    const fill = $('meter-fill');
    fill.style.width = `${(confidence * 100).toFixed(0)}%`;
    fill.style.background = confidence > 0.6 ? 'var(--ok)' : confidence > 0.3 ? 'var(--accent)' : 'var(--danger)';
  }

  setDebug(on: boolean): void {
    $('debug-btn').classList.toggle('on', on);
  }

  setStatus(text: string): void {
    $('status').textContent = text;
  }

  banner(title: string | null, sub = ''): void {
    const b = $('banner');
    b.hidden = title === null;
    $('banner-title').textContent = title ?? '';
    $('banner-sub').innerHTML = sub;
  }

  flashRecord(): void {
    const el = $('best');
    el.classList.remove('flash');
    void el.offsetWidth; // restart the CSS animation
    el.classList.add('flash');
  }
}
