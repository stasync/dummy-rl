"""Render MP4s of a controller taking hits.

    python scripts/render_video.py --run runs/A0_full                       # AI, max level reached
    python scripts/render_video.py --run runs/A0_full --controller stiff --level 4
    python scripts/render_video.py --run runs/A0_full --controller ai stiff limp --seed 3

Same seed => same hit sequence for every controller, so the clips are directly comparable
(Limp vs Stiff vs AI, PLAN.md §9.2). Physics randomization and obs noise are off: videos
show the nominal robot. Hits are drawn as red arrows (length ~ impulse).
Output: runs/<run>/videos/<controller>_L<level>_s<seed>.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stagger.config import load_config  # noqa: E402
from stagger.env import StaggerEnv  # noqa: E402
from stagger.robot import set_limp  # noqa: E402

ARROW_SECONDS = 0.3
ARROW_M_PER_NS = 0.025  # arrow length per N*s of impulse
TAIL_STEPS = 75         # keep filming 1.5 s after a knockdown


def find_model(run: Path) -> Path:
    if (run / "model.zip").exists():
        return run / "model.zip"
    ckpts = sorted((run / "checkpoints").glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        sys.exit(f"no model.zip or checkpoints in {run}")
    return ckpts[-1]


def add_arrow(scene: mujoco.MjvScene, start: np.ndarray, end: np.ndarray, rgba=(0.9, 0.1, 0.1, 1.0)) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.015, start, end)
    scene.ngeom += 1


def render_episode(env: StaggerEnv, policy, controller: str, seed: int, seconds: float, width: int, height: int) -> list[np.ndarray]:
    set_limp(env.model, controller == "limp")
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = env.model.camera("front").id
    obs, _ = env.reset(seed=seed)
    frames, arrows = [], []  # arrows: (start, end, steps_left)
    max_steps = int(seconds / env.dt)
    knocked_at = None
    for t in range(max_steps):
        if controller == "ai":
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape)  # stiff: hold the default pose; limp: ignored
        obs, _, terminated, _, info = env.step(action)
        for h in env.last_hits:
            # Arrow ends at the hit point, pointing along the impulse (where the bullet came from -> where it pushes).
            arrows.append((h.point - h.impulse * ARROW_M_PER_NS, h.point.copy(), int(ARROW_SECONDS / env.dt)))
        renderer.update_scene(env.data, camera=cam)
        for start, end, _ in arrows:
            add_arrow(renderer.scene, start, end)
        arrows = [(s, e, n - 1) for s, e, n in arrows if n > 1]
        frames.append(renderer.render().copy())
        if terminated and knocked_at is None:
            knocked_at = t
            print(f"  {controller}: knockdown at {t * env.dt:.2f}s ({info['knockdown']})")
        if knocked_at is not None and t - knocked_at >= TAIL_STEPS:
            break
    if knocked_at is None:
        print(f"  {controller}: survived {seconds:.0f}s")
    renderer.close()
    set_limp(env.model, False)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name> directory")
    ap.add_argument("--checkpoint", default=None, help="specific .zip (default: model.zip or latest checkpoint)")
    ap.add_argument("--controller", nargs="+", default=["ai"], choices=["ai", "stiff", "limp"])
    ap.add_argument("--level", type=int, default=None, help="hit curriculum level (default: final level of the run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"))
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    cfg.randomization.enabled = False
    env = StaggerEnv(cfg)

    level = args.level
    if level is None:
        summary = run / "summary.json"
        level = json.loads(summary.read_text())["final_curriculum_level"] if summary.exists() else cfg.curriculum.max_level
    env.set_level(level)

    policy = None
    if "ai" in args.controller:
        from stable_baselines3 import PPO

        path = Path(args.checkpoint) if args.checkpoint else find_model(run)
        print(f"policy: {path}")
        policy = PPO.load(path, device="cpu")

    out_dir = run / "videos"
    out_dir.mkdir(exist_ok=True)
    print(f"level {level} (J_max {env.schedule.j_max:.0f} N*s), seed {args.seed}")
    for c in args.controller:
        frames = render_episode(env, policy, c, args.seed, args.seconds, *args.size)
        out = out_dir / f"{c}_L{level}_s{args.seed}.mp4"
        imageio.mimsave(out, frames, fps=round(1 / env.dt), macro_block_size=8)
        print(f"  wrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
