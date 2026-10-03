# Experiments

Every run: config, steps, result, what I saw. Newest at the bottom of each day.
Machine: MacBook Pro M1 Pro (8P + 2E cores, 16 GB), CPU only. MuJoCo 3.14.0 (Python and WASM).

## Day 1 (2026-10-03)

### E0: robot PD gains (no learning)

- **Question:** do the plan's starting gains (hips/knees kp 150, ankles 60, arms 30) hold the default pose (Stiff mode) for 10 s?
- **Result:** no. Stiff tipped over backward in ~2 s. Knee sag under gravity load (5 N·m / kp 150 = 0.034 rad) tilts the pelvis back, which loads the knees more: a slow runaway. Separately, holding the body as an inverted pendulum over the ankles needs total ankle stiffness above m·g·h ≈ 30 · 9.81 · 0.75 ≈ 220 N·m/rad, and 2 × 60 = 120 is below that.
- **Sweep** (Stiff mode, 10 s, torso "up" component at the end; 1.0 = upright):

  | legs kp \ ankles kp | 150 | 250 | 400 |
  | --- | --- | --- | --- |
  | 150 | 0.07 (fell) | 0.991 (leans, drifts) | 0.996 |
  | 250 | 0.998 | 0.999 | 1.000 |
  | 400 | 1.000 | 1.000 | 1.000 |

- **Chosen:** legs kp 250 (forcerange ±90), ankles kp 200 (±45), arms 30 (±20). Stands at nominal gains and at −10% (the weakest motors under randomization). Covered by `tests/test_robot.py`.
- **Takeaway:** stiffer joints don't make the AI's job trivial. Stiff mode still survives only ~50% of episodes at level 2 (8 N·s) and none at level 5 (20 N·s), so most of the curriculum needs learned recovery.

### E0b: Stiff baseline vs curriculum level (no learning)

30 episodes per level, nominal physics, zero action:

| level | J_max (N·s) | Stiff survival | mean ep length (steps of 500) |
| --- | --- | --- | --- |
| 1 | 4 | 1.00 | 500 |
| 2 | 8 | 0.53 | 423 |
| 3 | 12 | 0.20 | 349 |
| 5 | 20 | 0.00 | 237 |
| 7 | 28 | 0.00 | 227 |
| 10 | 40 | 0.00 | 158 |

All knockdowns were `tilt` (> 60° from vertical, before any body part reached the floor).

### E0c: throughput

`scripts/bench.py`, zero action, SubprocVecEnv:

| envs | sim env-steps/s |
| --- | --- |
| 1 | 3,468 |
| 6 | 14,480 |
| 8 | **18,552** |
| 10 | 14,456 |

8 envs is the sweet spot (10 also hits the 2 efficiency cores and contends with the trainer process).
With PPO updates included, training runs at **~7,000 env-steps/s**, so 10M steps takes ~24 min and 30M takes ~70 min.
About 60% of wall time is the main process (policy inference per step + gradient updates), not physics.

### WASM check

`web/tests/smoke.test.ts`: the same XML in `@mujoco/mujoco` 3.14.0 under Node. Stiff stands, Limp collapses, and
after 500 physics steps from the same start, qpos matches native Python to **2.5e-16**. The parity target in PLAN.md §7 (1e-6) is very achievable.

### Hit-model check

`tests/test_env.py::test_hit_delivers_its_impulse`: in zero gravity, a hit changes total linear momentum by exactly J
(to 1e-6 with RK4). With the training integrator (implicitfast) it is 1–5% off, because the limbs swing during the
20 ms push and a first-order integrator doesn't conserve momentum exactly. That is integration error, not a hit-model
bug, and it is identical in Python and the browser.

### D1_short: first PPO run (curriculum from level 0)

- **Config:** `configs/base.yaml` (curriculum + randomization on), `--timesteps 3000000`, 8 envs.
- **Wall clock:** 444 s, **6,750 env-steps/s**.
- **Curriculum:** level 0 → 1 at 0.69M steps (first it had to learn to stand still through its own exploration noise;
  the untrained policy fell in ~90 steps with no hits at all), level 3 at 0.93M, then moved between 3–5, ending at **5** (J_max 20 N·s).
- **Eval** (nominal physics, deterministic policy, 40 episodes per level, same seeds for both controllers):

  | level | J_max (N·s) | Stiff survival | AI survival | AI knockdowns |
  | --- | --- | --- | --- | --- |
  | 2 | 8 | 0.60 | **1.00** | |
  | 3 | 12 | 0.20 | **0.97** | height 1 |
  | 4 | 16 | 0.05 | **0.90** | height 4 |
  | 5 | 20 | 0.00 | **0.65** | height 9, tilt 4, lower_arm_L 1 |
  | 6 | 24 | 0.00 | **0.33** | height 22, tilt 4, lower_arm_L 1 |
  | 8 | 32 | 0.00 | **0.07** | height 24, tilt 12, lower_arm_L 1 |

- **What I saw** (`runs/D1_short/videos/{ai,stiff,limp}_L5_s3.mp4`, same hits): AI survives the full 10 s and shifts its
  feet between hits (small recovery steps), arms move for balance. Stiff topples at 4.6 s, Limp at 0.7 s.
- **Open question for Day 2:** most AI knockdowns are `height` (pelvis < 55% of standing height), not `tilt`. It looks
  like it squats to absorb hits and sinks too low. Watch whether this persists in the long run; options are a stronger
  `height` weight or a slightly lower threshold.

### D1_overnight: 30M steps, full curriculum (running)

- **Config:** `configs/base.yaml`, 30M steps, 8 envs, launched 2026-10-03 ~15:40, ETA ~75 min.
