# CLAUDE.md: stagger

Browser game + RL portfolio: shoot a humanoid crash-test robot (raycast). A PPO policy trained in MuJoCo keeps it balanced. The game runs the same physics (MuJoCo WASM) and the same policy (plain TS) live in the browser. The full plan, specs and day-by-day tasks are in PLAN.md. Read it before starting work.

## Working agreement

- Work milestone by milestone (PLAN.md §8). At the end of each task, update Status below.
- The owner is new to RL. Explain why in short code comments. Keep code readable over clever.
- Training runs on the owner's Mac (CPU). Never assume a GPU. Use PyTorch device="cpu".
- Experiments are YAML configs in configs/. Don't hardcode hyperparameters, reward weights or hit-model constants.
- Every reward term must be a named function in stagger/rewards.py and be logged per term.
- Log every experiment (config, steps, result, what was observed) in EXPERIMENTS.md.
- Mirrored code must stay identical in behavior: obs.py ↔ obs.ts, hits.py ↔ hits.ts, falls.py ↔ falls.ts. Run the parity test (PLAN.md §7) after touching any of them or the export.
- web/public/policy.json is the contract between training and the game. Bump schema on any change.
- Everything must be original. No ARC Raiders names, assets or look-alike designs.
- Respect the cut list (PLAN.md §10). Don't add scope.

## Commands

```sh
source .venv/bin/activate
python scripts/train.py --config configs/base.yaml --name A0_full
tensorboard --logdir runs
python scripts/eval.py --runs runs/A0_full runs/A1_no_curriculum --out web/public/img
python scripts/render_video.py --run runs/A0_full
python scripts/export_policy.py --run runs/A0_full --out web/public/policy.json
python scripts/parity_fixtures.py --run runs/A0_full --out web/tests/fixtures.json
pytest
cd web && npm run dev        # the game
cd web && npx vitest run     # WASM smoke test (+ parity test, Day 2)
python scripts/bench.py --envs 1 8   # raw env-steps/s
python scripts/run_queue.py         # overnight: A0_full + ablations, 2 at a time
python scripts/eval.py --runs stiff runs/A0_full --levels 2 4 6 8   # survival + steps/hit
```

## Status

- [x] Day 1: it stands and takes small hits (D1_short: 97% survival at 12 N·s vs 20% for Stiff; first MP4s; 6.75k steps/s)
- [~] Day 2: tuning, export, robot in the browser. Export, parity (all pass, ~1e-7 / 2e-13) and the browser robot are done;
  reward/stepping tuning in progress (T1/T2 → action_scale 0.6; S1–S3 stepping experiments), overnight queue not started
- [~] Day 3: make it a game. Weapons, sparks/shake/hit-stop/sound, knockdown → slow-mo replay → rewind, best scores,
  balance meter, debug overlay (CoM, capture point, support polygon, torque tint, hit arrows) and look pass are written;
  needs a play-test in a real browser
- [ ] Day 4: analysis and ship

Notes / blockers:

- PD gains differ from the plan's first guesses (legs kp 250, ankles 200), because Stiff mode toppled otherwise. See EXPERIMENTS.md E0.
- `energy` reward weight was missing in the original plan doc; placeholder −0.0005 in configs/base.yaml.
- Throughput: use 8 envs on the M1 Pro (10 is slower). See EXPERIMENTS.md E0c.
- Obs and knockdown use only qpos/qvel + data.contact (no xmat/cvel) so the TS mirror needs no mj_forward tricks.
- MuJoCo pinned to 3.14.0 in both pyproject.toml and web/package.json; WASM vs native qpos diff is 2.5e-16 after 500 steps.
- D1_overnight plateaued at level ~4.3 and was stopped at 8.5M (EXPERIMENTS.md). Diagnosis: too little stepping;
  fixes = wider action range (done), capture-point reward + weapon-like hit patterns (S2/S3), step metric logged.
- Old run configs (curriculum.burst_*) are auto-migrated by config._migrate so they stay evaluable.
- Game keys: click shoot · Q/E/wheel weapon · 1/2/3 AI/Stiff/Limp · R rewind (after KO) · Enter new round · Tab (or HUD button) debug.
- web/public/policy.json = A0_full (30M steps, level 9, trained to 36 N·s): 53% survival at 40 N·s vs 0% Stiff.
