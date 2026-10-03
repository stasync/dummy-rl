"""Dump Python reference cases for the TypeScript parity test (PLAN.md §7).

    python scripts/parity_fixtures.py --run runs/A0_full --out web/tests/fixtures.json

Each case is a real state visited while the robot was being shot at (AI, Stiff and Limp
episodes, so knockdowns are included), plus:
  - obs, action, value: from build_obs and the SB3 policy (torch)
  - knockdown: FallDetector.check after mj_forward on that state
  - a hit, and qvel + knockdown after one control step with that hit applied

How a case is replayed (identically in Python and TS) -- every step matters for bit-parity:
  fresh MjData -> set qpos, qvel, ctrl -> mj_forward -> [check knockdown]
  -> applyHits -> mj_step x frame_skip -> clearHits -> [compare qvel, check knockdown]
A fresh MjData matters: MuJoCo's solver warm-starts from the previous step's accelerations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stagger.config import REPO_ROOT, load_config  # noqa: E402
from stagger.env import StaggerEnv  # noqa: E402
from stagger.hits import Hit, apply_hits, clear_hits  # noqa: E402
from stagger.obs import build_obs  # noqa: E402
from stagger.robot import set_limp  # noqa: E402


def collect_states(env: StaggerEnv, model: PPO, controller: str, n: int, rng: np.random.Generator, seed: int) -> list[dict]:
    """Play episodes and keep n random states (more weight near the end, where falls happen)."""
    set_limp(env.model, controller == "limp")
    states, ep = [], 0
    while len(states) < n:
        obs, _ = env.reset(seed=seed + ep)
        ep += 1
        def snapshot() -> dict:
            return {
                "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(), "prev_action": env.prev_action.copy(),
                "home_xy": env.home_xy.copy(), "home_yaw": env.home_yaw,
            }

        traj, done = [], False
        while not done:
            traj.append(snapshot())
            action = model.predict(obs, deterministic=True)[0] if controller == "ai" else np.zeros(18)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        traj.append(snapshot())  # the final state (knocked down, if the episode ended in a fall)
        picks = rng.choice(len(traj), size=min(3, len(traj)), replace=False).tolist() + [len(traj) - 1]
        states += [traj[i] for i in picks]
    set_limp(env.model, False)
    return states[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "web/tests/fixtures.json"))
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    cfg.randomization.enabled = False  # the game runs the nominal XML
    env = StaggerEnv(cfg)
    env.set_level(6)
    model = PPO.load(Path(args.checkpoint) if args.checkpoint else run / "model.zip", device="cpu")
    rng = np.random.default_rng(args.seed)

    n_ai = args.cases * 3 // 5
    n_other = (args.cases - n_ai) // 2
    states = (
        collect_states(env, model, "ai", n_ai, rng, 0)
        + collect_states(env, model, "stiff", n_other, rng, 1000)
        + collect_states(env, model, "limp", args.cases - n_ai - n_other, rng, 2000)
    )

    m, r, c = env.model, env.robot, cfg
    hit_bodies = list(range(1, m.nbody))
    cases = []
    for i, s in enumerate(states):
        obs = build_obs(s["qpos"], s["qvel"], s["prev_action"], r.default_joint_pos, s["home_xy"], s["home_yaw"], c.obs.scales, c.obs.clip)
        with torch.no_grad():
            t = torch.as_tensor(obs[None], dtype=torch.float32)
            action = np.clip(model.policy.get_distribution(t).mode().numpy()[0], -1.0, 1.0)
            value = float(model.policy.predict_values(t)[0, 0])
        # float64 like env.step (numpy would otherwise do 0.4 * float32 action in float32).
        ctrl = r.default_joint_pos + c.env.action_scale * action.astype(np.float64)

        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:], d.ctrl[:] = s["qpos"], s["qvel"], ctrl
        mujoco.mj_forward(m, d)
        knock_before = env.falls.check(d)

        # One hit per case: random body, point and impulse up to 40 N*s (beyond training on purpose).
        body = int(rng.choice(hit_bodies))
        g = r.body_main_geom(body)
        aabb = m.geom_aabb[g]
        point = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (aabb[:3] + rng.uniform(-1, 1, 3) * aabb[3:])
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        hit = Hit(body, point, direction * rng.uniform(1.0, 40.0))
        apply_hits(m, d, [hit], c.env.frame_skip)
        mujoco.mj_step(m, d, nstep=c.env.frame_skip)
        clear_hits(d)
        knock_after = env.falls.check(d)

        cases.append(
            {
                "qpos": s["qpos"].tolist(), "qvel": s["qvel"].tolist(), "prev_action": s["prev_action"].tolist(),
                "home_xy": s["home_xy"].tolist(), "home_yaw": s["home_yaw"],
                "obs": obs.astype(float).tolist(), "action": action.astype(float).tolist(), "value": value,
                "ctrl": ctrl.tolist(),
                "knockdown": knock_before,
                "hit": {"body_id": body, "point": hit.point.tolist(), "impulse": hit.impulse.tolist()},
                "qvel_after_hit": d.qvel.tolist(), "knockdown_after_hit": knock_after,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": run.name, "mujoco": mujoco.__version__, "cases": cases}))
    n_ko = sum(bool(x["knockdown"]) for x in cases)
    print(f"wrote {out}: {len(cases)} cases ({n_ko} knocked down before the hit, "
          f"{sum(bool(x['knockdown_after_hit']) for x in cases)} after)")


if __name__ == "__main__":
    main()
