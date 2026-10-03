# Stagger: shoot the robot, watch it keep its balance (4-day plan)

Product: a small browser game. A humanoid "crash-test robot" stands in a firing range. You click to shoot it (raycast). Every hit is a real physics impulse, and a neural network trained with reinforcement learning (RL) tries to keep the robot on its feet. There are no animations: every wobble, stumble, recovery step and fall comes from physics plus the learned controller. Knock it down in as few shots as possible.

Why: a real, publishable product (GitHub Pages + itch.io) that doubles as my portfolio for Gameplay AI / ML Engineer at Embark Studios (ARC Raiders), where the focus is ML locomotion for physics-driven robots that players shoot.

---

## 0. Context for Claude (read first)

- Owner: Stanislav. 10+ years in gameplay engineering (Unity/C#, DOTS/HPC#, shipped GTFO at 10 Chambers, 3+ years as lead). Currently full-stack TS/Node. New to hands-on RL, so explain RL decisions briefly in code comments and commit messages. He must be able to defend every choice in an interview.
- Time box: 4 days. Scope is fixed. Cut features before extending the timeline (§10).
- Training runs on the owner's MacBook (Apple Silicon, CPU). Keep everything CPU-friendly. PyTorch device="cpu".
- The game runs live in the browser: MuJoCo WASM for physics and the trained policy in plain TypeScript. So Python↔TS parity is critical (§7).
- The C++/Unreal side of the role is covered by the CV. This project covers the ML side, the tooling and game feel.
- Everything must be original. No ARC Raiders assets, names or look-alike designs. The brief: crash-test-dummy robot in an industrial firing range.

### What the job ad asks for, and where this project proves it

| Job ad item | Proof in this project |
| --- | --- |
| Research, develop, train RL models | PPO balance / push-recovery policy for an 18-DoF humanoid |
| Prototype RL locomotion platforms | Config-driven robot, task and reward setup |
| Model performance analyses & visualizations | Vulnerability heatmap, survival curves, ablations, live "balance confidence" meter from the critic |
| Improve game-side training framework | Per-term reward logging, one shared hit model used by training and the game, parity tests, in-game debug overlays |
| Reward functions (bonus) | Named reward terms plus an ablation study |
| Training curriculums (bonus) | Hit strength, location and frequency curriculum, plus a no-curriculum ablation |
| Emergent gameplay AI | Stumbles, recovery steps and arm flailing that nobody animated, all triggered by player shots |
| Speaks "design" | Weapon tuning against the training range, Limp/Stiff/AI modes, slow-mo, rewind, score loop |
| Python (bonus) | Whole training stack |

---

## 1. Deliverables

1. The game, playable in the browser: GitHub Pages, plus a free itch.io release.
2. "How it works" section under the game: short write-up, charts, comparison clips.
3. GitHub repo `stagger` (public): README, a reproducible training command, configs for every experiment shown.
4. Videos: 20–30 s hero clip (screen capture of the game) plus 3 short clips (Limp vs Stiff vs AI, bloopers, a big recovery).
5. Short application note (5–6 sentences) linking the game.

---

## 2. Tech stack

| Layer | Choice | Notes |
| --- | --- | --- |
| Physics | MuJoCo 3.x (Python `mujoco`) | The same XML is loaded by the browser |
| Env API | Gymnasium | `check_env` must pass |
| RL | Stable-Baselines3 PPO | SubprocVecEnv, N envs ≈ CPU cores |
| Logging | TensorBoard | Every reward term logged separately |
| Config | YAML (+ small dataclass loader) | Experiments = config files, no code edits |
| Plots | matplotlib | PNG/SVG for the page |
| Video | `mujoco.Renderer` + `imageio[ffmpeg]` | On macOS, if offscreen GL fails, try `MUJOCO_GL=cgl` or `glfw` |
| Game | Vite + TypeScript + Three.js + `@mujoco/mujoco` (single-threaded build) | Works on GitHub Pages and itch.io without special headers |
| Tests | pytest, vitest | vitest runs MuJoCo WASM in Node (verified working) |

Plan B for compute: if humanoid training on the Mac is too slow, move training to a free Colab GPU with MuJoCo's GPU version (MJX, via MuJoCo Playground / Brax PPO). Same XML, same rewards.

Setup (Mac):

```sh
uv venv -p 3.11 && source .venv/bin/activate
uv pip install mujoco gymnasium "stable-baselines3[extra]" torch tensorboard pyyaml matplotlib "imageio[ffmpeg]" pytest
# interactive MuJoCo viewer on macOS: mjpython -m mujoco.viewer --mjcf assets/stagger.xml
cd web && npm create vite@latest . -- --template vanilla-ts && npm i three @mujoco/mujoco && npm i -D @types/three vitest
```

---

## 3. Repo layout

```
stagger/
  CLAUDE.md                # conventions + current status (keep updated)
  PLAN.md                  # this file
  EXPERIMENTS.md           # every run: config, steps, result, what I saw
  README.md
  pyproject.toml
  assets/stagger.xml       # robot + range floor (primitives only)
  configs/
    base.yaml              # env, reward weights, hit model, PPO hyperparams
    ablations/{A1_no_curriculum,A2_no_home,A3_no_posture,A4_no_randomization}.yaml
  stagger/                 # python package
    robot.py               # model loading, joint/body indexing, default pose, hit regions
    env.py                 # StaggerEnv(gymnasium.Env)
    obs.py                 # observation builder      <-- MIRRORED in web/src/obs.ts
    hits.py                # hit model (impulse at a point)  <-- MIRRORED in web/src/hits.ts
    falls.py               # knockdown detection       <-- MIRRORED in web/src/falls.ts
    rewards.py             # reward-term registry (name -> fn), weights from config
    curriculum.py          # hit strength / location / frequency
    randomization.py       # friction, mass, motor strength, latency, obs noise
  scripts/
    train.py               # python scripts/train.py --config configs/base.yaml --name A0_full
    eval.py                # survival sweeps -> metrics.json + plots
    render_video.py
    export_policy.py       # SB3 -> web/public/policy.json (actor + critic)
    parity_fixtures.py     # dumps cases for the TS parity test
    plots.py
  runs/                    # gitignored
  web/
    src/{main.ts, scene.ts, sim.ts, policy.ts, obs.ts, hits.ts, falls.ts, weapons.ts, game.ts, hud.ts, debug.ts, audio.ts}
    public/{assets/stagger.xml, policy.json, img/, videos/}
    tests/parity.test.ts
  .github/workflows/pages.yml
```

---

## 4. Environment spec

### 4.1 Robot: crash-test humanoid

- Toy-like proportions: about 1.3 m tall, about 30 kg, slightly oversized head, big flat feet (which make balance much easier).
- Primitive geoms only (capsules, boxes, spheres). Named bodies so hits can be grouped into regions: head, torso, pelvis, upper_arm_L/R, lower_arm_L/R, thigh_L/R, shin_L/R, foot_L/R.
- 18 actuated joints:
  - legs (×2): hip yaw, hip roll, hip pitch, knee, ankle pitch, ankle roll = 12
  - arms (×2): shoulder pitch, shoulder roll, elbow = 6
  - head fixed to torso, no waist joint (v1)
- `<position>` actuators (PD) with torque limits (forcerange). Starting points to tune: hips/knees kp ≈ 150, ankles ≈ 60, arms ≈ 30; joint damping and armature set.
- Sim timestep 0.002 s, policy at 50 Hz (frame_skip 10).
- Acceptance:
  - In Stiff mode (hold default pose) it stands for 10 s.
  - In Limp mode it collapses like a puppet with cut strings.
  - The XML loads in Python and in `@mujoco/mujoco` in Node.
- If learning is too slow: freeze hip yaw first, then make the arms passive (damped, unactuated).

### 4.2 Action

- 18 values in [-1, 1] (clipped) → target = default_qpos + action_scale * a, with action_scale ≈ 0.4 rad (tune).

### 4.3 Observation (fixed scales, no VecNormalize, which keeps TS parity simple)

The policy does not see the bullets. It only feels them through its own body, the way a real robot would.

| Part | Dim | Notes |
| --- | --- | --- |
| base angular velocity (body frame) | 3 | Freejoint qvel[3:6] is already body frame |
| projected gravity (body frame) | 3 | "Which way is down" |
| base linear velocity (body frame) | 3 | Freejoint qvel[0:3] is world frame, so rotate it |
| offset to home spot (heading frame) | 2 | Lets it step back to where it stood |
| heading error (sin, cos) | 2 | Lets it turn back to face the range |
| joint pos − default | 18 | |
| joint vel (scaled) | 18 | |
| previous action | 18 | |
| total | 67 | Scales live in config and are exported to policy.json |

### 4.4 Hit model (hits.py), shared exactly with the game

- A hit = (body_id, world_point, impulse_vector J).
- Applied as a force for one control step (10 substeps): F = J / (frame_skip * timestep). Set `xfrc_applied[body] = [F, (point − xipos[body]) × F]`, then clear it after the step.
- In training, hits are sampled: region (weighted: torso 35%, pelvis 15%, head 15%, arms 15%, legs 20%), a random point on that body, a mostly horizontal direction from a random azimuth, and a magnitude from the curriculum.
- In the game, the same function is called with the raycast hit point and the weapon's impulse.

### 4.5 Episodes & knockdown (falls.py, shared with the game)

- 10 s episodes (500 policy steps). 0.5 s settle, then hits every 0.8–2.5 s.
- Knockdown = head, torso, pelvis, arms or thighs touching the floor, or pelvis height < 55% of standing height, or torso tilt > 60°. Ends the episode, and means a KO in the game.

### 4.6 Reward terms (starting weights, tune on Day 1–2)

Every term is a named function in rewards.py, weighted via YAML, and logged per term.

| Term | Formula | Weight |
| --- | --- | --- |
| alive | 1 per step | +1.0 |
| upright | exp(-‖g_xy‖² / 0.05) | +1.0 |
| height | exp(-(h_pelvis − h0)² / 0.005) | +0.5 |
| home | exp(-‖p_xy − home‖² / 0.5) | +0.5 |
| heading | exp(-yaw_err² / 0.3) | +0.2 |
| calm | exp(-‖v_base‖² / 0.1) | +0.3 |
| posture | ‖q − q_default‖² | −0.1 |
| energy | Σ\|τ · q̇\| | −0.0005 (*weight missing in the original doc; placeholder, tune Day 1–2*) |
| action_rate | ‖a_t − a_{t−1}‖² | −0.02 |
| joint_limit | sum of limit violations | −1.0 |
| foot_slip | Σ_in-contact ‖v_foot,xy‖² | −0.1 |
| termination | on knockdown | −20.0 |

What to look for: too much posture/calm makes it stiff (topples like a statue), and too little makes it sloppy. Stepping should appear once hits get big enough that ankles and hips can't absorb them.

### 4.7 Curriculum (curriculum.py)

One global level, set from a callback via `env_method("set_level", L)`.

1. Hit strength: J_max ramps from 0 to about 40 N·s over levels 0–10. Promote when survival over the last N episodes > 80%. Demote when < 50%.
2. Hit location: start with torso and pelvis only (near the center of mass). Add head, arms and legs from level 3.
3. Hit frequency: the interval shrinks from 2.5 s to 0.8 s. From level 6, occasional bursts (shotgun-like multi-hits).

### 4.8 Domain randomization

Friction 0.6–1.2, body masses ±10%, motor strength and PD gains ±10%, small obs noise, 0–1 step action latency (helps the policy survive browser timing jitter).

### 4.9 PPO starting point (SB3)

n_envs = cpu_count, n_steps = 256 per env, batch_size ≈ n_envs*n_steps/4, n_epochs = 5, lr = 3e-4 (linear decay), gamma = 0.99, gae_lambda = 0.95, clip = 0.2, ent_coef = 0.0–0.005, separate actor and critic nets [256, 256, 128], ELU, log_std_init = −1. Budget: 20–50M steps for the main run, 10M per ablation (same seed and budget). Measure env-steps/s on Day 1.

---

## 5. Evaluation (scripts/eval.py → plots.py)

Fixed seeds, deterministic policy. Compares Limp, Stiff (PD holds the default pose), AI, and the ablations.

| Analysis | Output |
| --- | --- |
| Survival vs hit strength (per controller) | Line chart. The headline result |
| Vulnerability map: region × direction (front/back/left/right) at a fixed strength | Heatmap on a body silhouette, plus "max survivable impulse" per region |
| Recovery | time to settle, recovery steps taken, drift from home |
| Ablations | learning curves + survival curves (A1 no curriculum, A2 no home, A3 no posture/calm, A4 no randomization) |
| Critic as a fall predictor | Does a low value V(s) predict a knockdown in the next second? ROC curve. Justifies the in-game meter |

---

## 6. The game (web/)

**Scene & look**

- Load stagger.xml with `@mujoco/mujoco`, build Three.js meshes from the geoms, update them from geom_xpos/geom_xmat.
- Industrial firing range: concrete floor, hanging lights, fog, and a crash-test robot in matte white and orange with target markings. Original look only.
- Fixed-timestep simulation loop (accumulator). Policy at 50 Hz, physics at 500 Hz.

**Core loop**

- Crosshair + click → Three.js raycast against the robot meshes → body id, hit point, ray direction → `applyHit()` from hits.ts.
- Weapons (design knobs, tuned against the training range J_max):
  - Pistol: inside the training range
  - Rifle: auto-fire
  - Shotgun: 6 pellets
  - Cannon: deliberately beyond the training range, for spectacular falls
- Knockdown (from falls.ts) → "KNOCKDOWN in N shots" → slow-mo replay of the last 3 s from a ring buffer → rewind to before the fatal shot (R), or reset.
- Score: fewest shots to knock it down, per weapon. Best scores go in localStorage, wrapped in try/catch.

**Modes** (keys 1/2/3): AI / Stiff / Limp. Switching shows instantly what the learned controller adds. This is the key portfolio moment.

**HUD & debug**

- Balance confidence meter: the critic's V(s), normalized with the min/max from eval. It drops when the robot is in trouble.
- Debug overlay (~ key): center of mass projected on the ground, support polygon, foot contacts, limbs tinted by torque/limit, hit arrows.
- Game feel: hit spark, a few frames of hit-stop on big hits, light camera shake, small synthesized WebAudio sounds (no audio assets).

**policy.json** (from export_policy.py), the contract between training and the game:

```json
{ "schema": 1, "obs_dim": 67, "act_dim": 18,
  "actor":  [{ "W": [], "b": [], "act": "elu" }, { "W": [], "b": [], "act": "none" }],
  "critic": [{ "W": [], "b": [], "act": "elu" }, { "W": [], "b": [], "act": "none" }],
  "value_range": [0.0, 1.0],
  "obs_scales": {}, "default_qpos": [], "action_scale": 0.4,
  "timestep": 0.002, "frame_skip": 10, "clip_actions": 1.0,
  "hit_model": { "j_max_trained": 40.0 } }
```

Deploy: a GitHub Action builds web/ to Pages. A zip of dist/ goes to itch.io as an HTML5 game.

---

## 7. Parity (don't skip)

parity_fixtures.py dumps 50 random states. tests/parity.test.ts loads each into MjData and checks:

- obs, action and value match Python (max abs diff < 1e-4)
- applyHit gives the same qvel after one control step as Python (< 1e-6)
- knockdown detection gives the same result

Run it after touching obs, hits, falls or the export.

---

## 8. Day-by-day

Launch long training runs before sleep or work. Keep the laptop plugged in and awake (`caffeinate -dimsu`). While training runs, Claude builds the game side.

### Day 1: it stands and takes small hits

- Repo scaffold, CLAUDE.md, configs, uv env (0.5 h)
- stagger.xml robot. Stiff stands, Limp collapses. WASM smoke test in Node (2 h)
- env.py, obs.py, hits.py, falls.py, rewards.py. check_env passes (2 h)
- train.py with SubprocVecEnv + per-term TensorBoard callback. Measure steps/s (1 h)
- render_video.py (0.5 h)
- Short run (standing only, then small torso hits), then overnight run with the curriculum
- **Done when:** it stands and absorbs small hits, the first MP4 exists, and steps/s is known.

### Day 2: tuning, export, robot in the browser

- Review the overnight run, tune rewards, log everything in EXPERIMENTS.md (1.5 h)
- Full curriculum (location, frequency) + domain randomization (1.5 h)
- export_policy.py (actor + critic) + parity fixtures + parity test (2 h)
- Web: scene from the XML, sim loop, policy running live, Stiff/Limp/AI modes (2.5 h)
- Overnight: main run (long) + ablations queued
- **Done when:** the trained robot stands in the browser, reacts to a debug shove, and parity passes.

### Day 3: make it a game

- Raycast shooting + applyHit (1.5 h)
- Weapons + game feel (spark, hit-stop, shake, sound) (1.5 h)
- Knockdown → slow-mo replay → rewind/reset, score + best scores (1.5 h)
- Balance meter, debug overlay (1.5 h)
- Look pass: range, lighting, robot materials (1 h)
- **Done when:** someone who has never seen it can play it and laugh.

### Day 4: analysis and ship

- Swap in the final policy, re-run parity (0.5 h)
- eval.py + plots.py: survival curves, vulnerability heatmap, ablations, critic ROC (2 h)
- "How it works" section (§9) (2 h)
- Hero video + 3 clips (screen capture) (1 h)
- Deploy to GitHub Pages + itch.io, README (0.5 h)
- Application note + send (0.5 h)

---

## 9. "How it works" section (under the game)

1. Pitch: "A crash-test robot that learned to keep its balance. Shoot it and watch it recover. No animations: every stumble is physics plus a neural network trained with reinforcement learning."
2. Limp vs Stiff vs AI: three short clips, same shots. The core visual.
3. The setup: what the robot feels (no bullets), what it controls, 50 Hz PD control (one diagram).
4. Reward design: term table plus what removing terms did (ablation clips + curves). Honest bloopers.
5. Curriculum: how the hits ramp up (level-over-time chart) and what happened without it.
6. Results: survival vs hit strength, vulnerability heatmap, critic as fall predictor.
7. Game design notes: weapons tuned against the training range, why the cannon is out of range on purpose, rewind and slow-mo.
8. How I'd take this to a game like ARC Raiders / Unreal: the policy in-engine via ONNX / Unreal NNE, a behavior tree on top, physics-parity notes, training-framework ideas (shared hit model, per-term dashboards, regression evals).
9. Links: repo, CV, LinkedIn. Note that Claude Code was used as a pair programmer, while the experiments and decisions are mine.

---

## 10. Cut list (drop in this order if behind)

1. itch.io release (GitHub Pages only)
2. Extra weapons, so a single gun only
3. Slow-mo replay and rewind, so a plain reset only
4. Balance meter and critic ROC
5. Ablations down to just A1 (no curriculum)
6. home / heading terms (just stay upright)
7. Actuated arms, so passive arms instead

Never cut: policy running live in the browser, shooting, Limp/Stiff/AI modes, the survival-vs-strength chart, per-term reward logging, the short write-up, the hero video.

---

## 11. Interview prep (fill in answers as we go)

- Why PPO (vs SAC)? Why exp-kernel rewards?
- The robot never sees the bullet. How does it still recover?
- Ankle vs hip vs stepping strategy: when did stepping appear in training?
- Stiffness vs compliance trade-off in the reward. What did too much posture do?
- What did the run without a curriculum do, and why?
- What does the balance meter actually show (critic V(s)), and how reliable is it?
- Sim-to-engine: MuJoCo vs Chaos/PhysX. What breaks and how to mitigate it?
- How would you make the training framework easier for designers (shared hit model, regression evals)?
- Compute budget: steps/s, wall-clock per run, what you'd scale with a GPU.

---

## 12. Product ladder (after the application)

- v2: a learned get-up policy (an endless loop with no resets) and a walking target (the policy also takes speed/turn commands; W/S/A/D control).
- v3: several robots, robots fighting each other, a player-controlled robot.
- Later, maybe: a toolkit for game devs (shared hit model, reward templates, web debugger, engine runtime). Keep training and game decoupled and configs clean so this stays possible. In the application, frame it as a tooling mindset, not a startup plan.
