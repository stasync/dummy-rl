"""Raw simulation throughput (no learning), to size n_envs and estimate run wall-clock.

    python scripts/bench.py --config configs/base.yaml --envs 1 6 8 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stagger.config import load_config  # noqa: E402
from stagger.env import make_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 8, 10])
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for n in args.envs:
        env = SubprocVecEnv([make_env(cfg) for _ in range(n)])
        env.seed(0)
        env.reset()
        acts = np.zeros((n, env.action_space.shape[0]), dtype=np.float32)  # Stiff: long episodes
        t0, steps = time.perf_counter(), 0
        while time.perf_counter() - t0 < args.seconds:
            env.step(acts)
            steps += n
        print(f"{n:3d} envs: {steps / (time.perf_counter() - t0):8,.0f} env-steps/s (sim only)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
