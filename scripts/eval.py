"""Evaluate controllers under identical hits: survival and recovery stepping per curriculum level.

    python scripts/eval.py --runs stiff runs/A0_full runs/D1_overnight@ppo_6000000_steps --levels 2 4 6 8
    python scripts/eval.py --runs stiff runs/A0_full --out web/public/img      # also writes metrics.json

A run spec is `runs/<name>` (model.zip, else the latest checkpoint), `runs/<name>@<checkpoint>`,
or `stiff` / `limp` (no policy). Each policy runs with its own env settings (action scale, obs),
but every controller faces the *same* hits: hit model + curriculum come from --hits-from, and
episode seeds are shared. Physics randomization and obs noise are off; the policy is deterministic.

Metrics per (controller, level): survival (episodes reaching 10 s), recovery steps per hit event
(a foot lifting off and landing >= 5 cm away), knockdown reasons.
Day 4 adds: survival vs exact hit strength, vulnerability map, critic ROC (PLAN.md §5).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stagger.config import REPO_ROOT, load_config  # noqa: E402


def resolve(spec: str) -> tuple[str, Path | None, Path | None]:
    """spec -> (label, run dir, checkpoint)."""
    if spec in ("stiff", "limp"):
        return spec, None, None
    run_s, _, ckpt = spec.partition("@")
    run = Path(run_s)
    if ckpt:
        path = run / "checkpoints" / f"{ckpt}.zip"
    elif (run / "model.zip").exists():
        path = run / "model.zip"
    else:
        ckpts = sorted((run / "checkpoints").glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
        path = ckpts[-1]
    return f"{run.name}@{path.stem}", run, path


def evaluate(job: tuple[str, str, int, int, int]) -> dict:
    spec, hits_from, level, episodes, seed = job
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO

    from stagger.env import StaggerEnv
    from stagger.robot import set_limp

    label, run, ckpt = resolve(spec)
    ref = load_config(hits_from)
    cfg = load_config(run / "config.yaml") if run else ref
    cfg.hits, cfg.curriculum = ref.hits, ref.curriculum  # same hits for every controller
    cfg.randomization.enabled = False
    env = StaggerEnv(cfg)
    env.set_level(level)
    set_limp(env.model, spec == "limp")
    policy = PPO.load(ckpt, device="cpu") if ckpt else None

    survived, steps, events, reasons, lengths = 0, 0, 0, Counter(), []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = policy.predict(obs, deterministic=True)[0] if policy else np.zeros(env.action_space.shape)
            obs, _, term, trunc, info = env.step(action)
            steps += info["steps_taken"]
            events += info["hit_events"]
            done = term or trunc
        survived += int(trunc)
        lengths.append(env.step_count)
        if term:
            reasons[info["knockdown"]] += 1
    return {
        "controller": label, "level": level, "j_max": env.schedule.j_max, "episodes": episodes,
        "survival": survived / episodes, "steps_per_event": steps / max(events, 1),
        "mean_len": float(np.mean(lengths)), "knockdowns": dict(reasons),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--hits-from", default=str(REPO_ROOT / "configs/base.yaml"))
    ap.add_argument("--seed", type=int, default=20_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="directory for metrics.json")
    args = ap.parse_args()

    jobs = [(spec, args.hits_from, lv, args.episodes, args.seed) for spec in args.runs for lv in args.levels]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(evaluate, jobs))

    print(f"{'controller':38s} {'level':>5s} {'J_max':>6s} {'survival':>8s} {'steps/hit':>9s}  knockdowns")
    for r in results:
        ko = ", ".join(f"{k} {v}" for k, v in sorted(r["knockdowns"].items(), key=lambda kv: -kv[1]))
        print(f"{r['controller']:38s} {r['level']:5d} {r['j_max']:6.0f} {r['survival']:8.2f} {r['steps_per_event']:9.2f}  {ko}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({"hits_from": args.hits_from, "results": results}, indent=1))
        print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
