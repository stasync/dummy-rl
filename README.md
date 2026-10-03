# Stagger

Shoot a crash-test robot and watch it keep its balance. There are no animations: every
wobble, stumble and recovery step comes from physics (MuJoCo) plus a neural network
trained with reinforcement learning (PPO). The game runs the same physics and the same
policy live in the browser.

> Work in progress. See [PLAN.md](PLAN.md) for the full plan and [EXPERIMENTS.md](EXPERIMENTS.md) for the training log.

## Quick start (training side, macOS / Linux, CPU only)

```sh
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -e .
pytest                                                    # robot + env sanity checks
python scripts/bench.py --envs 1 8                        # env-steps/s on this machine
python scripts/train.py --config configs/base.yaml --name A0_full
tensorboard --logdir runs                                 # per-term rewards under rew/*
python scripts/render_video.py --run runs/A0_full --controller ai stiff limp
```

## Web side

```sh
cd web && npm install
npx vitest run     # MuJoCo WASM smoke test (+ parity test, later)
```

## Layout

- `assets/stagger.xml`: robot + floor, primitives only. Loaded by both Python and the browser.
- `configs/`: experiments are YAML files; ablations inherit `base.yaml`.
- `stagger/`: env, observation, hit model, knockdown detection, reward terms, curriculum, randomization.
- `scripts/`: train, render video, benchmark (eval / export / parity to come).
- `web/`: the game (Vite + TypeScript + Three.js + `@mujoco/mujoco`).
