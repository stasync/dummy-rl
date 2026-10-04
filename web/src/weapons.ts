// Weapons are design knobs, tuned against the strongest hit the policy trained on (j_max_trained).
// Inside that range the robot has a real chance to recover; the cannon is deliberately outside
// it, so it makes spectacular falls no matter how good the controller is.

export type WeaponId = 'pistol' | 'rifle' | 'shotgun' | 'cannon';

export interface Weapon {
  id: WeaponId;
  name: string;
  impulse: number;   // N*s per projectile
  pellets: number;   // projectiles per shot
  spreadDeg: number; // cone half-angle
  cooldown: number;  // seconds between shots
  auto: boolean;     // keeps firing while the button is held
  note: string;      // shown in the HUD
}

/** Fractions of j_max_trained. Training saw singles up to 1.0x and multi-hit events up to 1.5x in total. */
export function makeWeapons(jTrained: number): Weapon[] {
  const J = (f: number) => Math.round(f * jTrained * 10) / 10;
  return [
    { id: 'pistol', name: 'Pistol', impulse: J(0.7), pellets: 1, spreadDeg: 0, cooldown: 0.25, auto: false, note: 'inside the training range' },
    { id: 'rifle', name: 'Rifle', impulse: J(0.2), pellets: 1, spreadDeg: 1.2, cooldown: 0.1, auto: true, note: 'auto-fire, like training rapid fire' },
    { id: 'shotgun', name: 'Shotgun', impulse: J(0.25), pellets: 6, spreadDeg: 4, cooldown: 0.9, auto: false, note: '6 pellets, like a training burst' },
    { id: 'cannon', name: 'Cannon', impulse: J(2.5), pellets: 1, spreadDeg: 0, cooldown: 1.5, auto: false, note: 'beyond the training range on purpose' },
  ];
}
