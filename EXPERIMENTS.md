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

### D1_overnight: full curriculum, stopped at 8.5M of 30M (plateau)

- **Config:** `configs/base.yaml`, 8 envs, ~6.2–7k env-steps/s. Stopped by hand at 8.5M steps, after it
  had been flat since ~3M; checkpoints at 2/4/6/8M (no model.zip, since SIGINT is ignored by `&` background jobs;
  train.py now saves on SIGINT *and* SIGTERM).
- **Curriculum:** reached level 3 by 1M, then bounced between 3–5 (98 level changes by 7M); mean level per 1M-step bin:
  0.7, 3.1, 3.1, 3.9, 4.3, 4.3, 4.2. Rollout survival flat at ~0.66.
- **Reward terms** (per-episode sums, 0–1M → 6–7M): energy −166 → −58, action_rate −32 → −19, foot_slip −21 → −6,
  calm +27 → +80. It got smoother and calmer, but not more capable. Action std shrank 0.32 → 0.17.
- **Knockdown analysis** (6M checkpoint, level 6 = 24 N·s, 60 episodes, nominal physics): survival **0.40**.
  - Reasons: tilt 30, height 5, arm contact 1.
  - Body that took the last hit before a KO: torso 19 (of 97 torso hits, 20%), **head 10 (of 29 head hits, 34%)**, others 1 each.
  - Push direction (relative to facing): left 12, right 9, forward 10, backward 5.
  - **Median 1.1 s from the last hit to the KO.** It doesn't topple instantly; it tries to recover and fails.
- **Diagnosis:** the action range limits stepping. With `action_scale` 0.4 rad the furthest a foot can be placed is
  ~0.11 m forward with ~4 cm clearance (hip −0.55, knee 0.7). The capture point for 24 N·s at CoM height ~0.73 m
  is v/ω₀ = (24/30) / √(9.81/0.73) ≈ 0.22 m, so it needs a ~0.2 m step. Reward weights look unlikely to be the blocker:
  posture/energy for a recovery step cost a few reward points, versus ~2 points per step for every step lost to a fall.
- **Next:** T1/T2 widen the action range (0.6 / 0.8 rad) with the same initial exploration in radians.

## Day 2 (2026-10-03)

### Parity (PLAN.md §7): passes

`scripts/export_policy.py` + `scripts/parity_fixtures.py` on D1_short, `web/tests/parity.test.ts`, 50 cases from AI/Stiff/Limp
episodes at level 6 (7 knocked down):

| check | max diff | target |
| --- | --- | --- |
| obs | 1.2e-7 | 1e-4 |
| action | 1.5e-7 | 1e-4 |
| value (relative) | 4.0e-7 | 1e-5 |
| qvel after applyHit + one control step | 1.9e-13 | 1e-6 |
| knockdown reason, before and after the hit | identical (50/50) | identical |

The obs difference is just Python's final float32 cast. The game's `Sim` (headless, `web/tests/sim.test.ts`): a 15 N·s
chest shove from the shooter's side, AI recovers (21 cm drift, meter dips to 0.87), Stiff topples (tilt).

### T1_action06 / T2_action08: wider action range

- `configs/tuning/T1_action06.yaml` (action_scale 0.6, log_std_init −1.4) and `T2_action08.yaml` (0.8, −1.69), 6M steps each,
  run in parallel (~4.8k env-steps/s each, ~9.5k combined: two runs at once use the CPU better than one).
  log_std_init keeps the initial exploration noise in *radians* equal to base (0.15 rad).
- Final curriculum level: T1 5, T2 6 (D1_overnight: bouncing 4–5).
- **Eval** (`scripts/eval.py`, 40 episodes per cell, same seeds; every controller faces the *new* weapon-pattern hits
  from base.yaml, so this also tests generalization to bursts/rapid fire none of them trained on much):

  | level (J_max) | Stiff | D1_overnight @6M (0.4) | T1 (0.6) | T2 (0.8) |
  | --- | --- | --- | --- | --- |
  | 3 (12 N·s) | 0.42 | 0.95 | 0.97 | 0.97 |
  | 5 (20 N·s) | 0.10 | 0.60 | **0.72** | **0.78** |
  | 7 (28 N·s) | 0.03 | 0.20 | **0.42** | 0.30 |

  Recovery steps per hit event at level 7: D1 0.97, T1 1.21, T2 1.02 (Stiff 0.30 = feet sliding while it falls).
- **Conclusion:** a wider action range helps (+12–22 points at the hard levels, same step budget). 0.6 vs 0.8 is within
  noise (±0.07 at n=40). **Chose 0.6** for base: smaller residual = smoother motion, less extreme PD targets in the game.

### Why it barely steps, and what to change

PPO settled in a local optimum: absorb hits with ankles/hips. Stepping is all-or-nothing. A half-step (lifting a foot)
shrinks the support area and makes a fall *more* likely, so exploration toward stepping is punished before a complete,
well-timed step is ever rewarded, and Gaussian action noise almost never produces one. Multi-hit events (bursts) were
also only trained from level 6, which the curriculum rarely reached.

Changes (each is a config switch, so it can be ablated):

1. **Measure it:** env counts recovery steps (foot lifts off, lands ≥ 5 cm away); logged as `ep/steps_per_hit_event`.
2. **Capture-point reward** (`capture` term): CP = CoM_xy + v_CoM_xy / √(g/h_CoM). Reward exp(−d²/0.01), with d = distance
   from the CP to the segment between the feet, minus 6 cm. Standing still it is ~1. After a big push the only ways back
   are braking or moving a foot toward the CP, and that pays off *during* the step, which gives PPO a gradient toward stepping.
3. **Weapon-like hit patterns from level 2:** single (60%), burst/shotgun (20%: 3–6 hits in one step), rapid/rifle
   (20%: 3–6 hits 0.1 s apart). Multi-hit events carry 1.5× the impulse, split evenly.
4. **Exploration:** entropy bonus 0.005 as a separate factor.

### S1 / S2 / S3: stepping experiments → capture-point reward wins

All on the new base (action_scale 0.6, hit patterns), 6M steps, same seed, 3 in parallel
(~1.9–3.8k env-steps/s each; the machine was also busy). S1 control (capture 0), S2 capture 1.0, S3 capture 1.0 + ent_coef 0.005.

- Final curriculum level: S1 **5**, S2 **7** (touched 8), S3 **7**.
- **Eval** (`scripts/eval.py`, 40 episodes per cell, same seeds, base.yaml hit patterns for everyone):

  | level (J_max) | Stiff | T1 (old hits) | S1 control | **S2 capture** | S3 capture + entropy |
  | --- | --- | --- | --- | --- | --- |
  | 3 (12 N·s) | 0.42 | 0.97 | 0.95 | 0.97 | 0.93 |
  | 5 (20 N·s) | 0.10 | 0.72 | 0.75 | **0.93** | 0.75 |
  | 7 (28 N·s) | 0.03 | 0.42 | 0.23 | **0.47** | 0.45 |
  | 9 (36 N·s) | 0.00 | 0.20 | 0.07 | **0.30** | 0.17 |
  | recovery steps / hit event @ L9 | 0.48* | 1.49 | 1.21 | **1.99** | 1.92 |

  \* Stiff can't step: its count is feet sliding while it topples (the metric's noise floor).
- **Conclusion:** capture-point shaping is the biggest single improvement so far. At the same budget it roughly
  quadruples survival at 36 N·s (0.07 → 0.30) and raises stepping from ~1.2 to ~2 steps per hit event. The entropy bonus
  adds nothing (worse at L5/L9, more `height` knockdowns). **base.yaml now has `capture: 1.0`, ent_coef 0.**
- Video: `runs/S2_capture/videos/{ai,stiff}_L7_s4.mp4`. AI survives 10 s at 28 N·s with shifting stance; Stiff topples at 3.6 s.
- Remaining failure mode at high levels is `height` (pelvis < 55% of standing height): legs fold instead of tipping over.
  To investigate after the main run.
- **Game:** `web/public/policy.json` = S2_capture (6M steps, level 7, j_max_trained 28 N·s). Parity: obs 1.2e-7,
  action 1.3e-7, value 1.2e-6 (relative), qvel after hit 3.0e-13, knockdown 50/50 (8 knocked down).
