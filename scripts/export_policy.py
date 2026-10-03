"""Export a trained SB3 policy to web/public/policy.json: the contract between training and the game.

    python scripts/export_policy.py --run runs/A0_full --out web/public/policy.json

Contains the actor (obs -> action mean) and the critic (obs -> V(s), for the balance meter),
plus every constant the browser needs to reproduce training exactly: obs scales, default
pose, action scale, timing, knockdown thresholds and the strongest hit the policy was trained on.
A sha256 of stagger.xml is included so the game can refuse a policy trained on a different robot.

Schema changes MUST bump SCHEMA (and web/src/policy.ts).

Layers are {"in", "out", "W" (row-major out x in), "b", "act"}. Weights are written as the
shortest decimal that round-trips to the same float32, so Float32Array in the browser holds
exactly torch's weights.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stagger.config import REPO_ROOT, load_config  # noqa: E402
from stagger.curriculum import schedule_for_level  # noqa: E402
from stagger.env import StaggerEnv  # noqa: E402
from stagger.robot import REGIONS  # noqa: E402

SCHEMA = 1
ACT_NAMES = {torch.nn.ELU: "elu", torch.nn.ReLU: "relu", torch.nn.Tanh: "tanh"}


def f32_list(a: np.ndarray) -> list[float]:
    return [float(np.format_float_positional(x, unique=True, trim="-")) for x in np.asarray(a, np.float32).ravel()]


def export_mlp(hidden: torch.nn.Sequential, head: torch.nn.Linear) -> list[dict]:
    layers, mods = [], list(hidden) + [head]
    i = 0
    while i < len(mods):
        lin = mods[i]
        assert isinstance(lin, torch.nn.Linear), f"unexpected module {lin}"
        act = "none"
        if i + 1 < len(mods) and type(mods[i + 1]) in ACT_NAMES:
            act = ACT_NAMES[type(mods[i + 1])]
            i += 1
        W = lin.weight.detach().cpu().numpy()
        layers.append({"in": W.shape[1], "out": W.shape[0], "W": f32_list(W), "b": f32_list(lin.bias.detach().cpu().numpy()), "act": act})
        i += 1
    return layers


def value_range(model: PPO, env: StaggerEnv, episodes: int, seed: int) -> list[float]:
    """1st/99th percentile of V(s) over rollouts at the trained level (falls included).
    Placeholder for the Day 4 eval-based range; good enough to normalize the HUD meter."""
    values = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            with torch.no_grad():
                v = model.policy.predict_values(torch.as_tensor(obs[None], dtype=torch.float32))
            values.append(float(v))
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    return [round(float(np.percentile(values, 1)), 4), round(float(np.percentile(values, 99)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None, help="default: <run>/model.zip")
    ap.add_argument("--out", default=str(REPO_ROOT / "web/public/policy.json"))
    ap.add_argument("--value-episodes", type=int, default=20)
    args = ap.parse_args()

    run = Path(args.run)
    ckpt = Path(args.checkpoint) if args.checkpoint else run / "model.zip"
    cfg = load_config(run / "config.yaml")
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    level = summary.get("final_curriculum_level", cfg.curriculum.max_level)
    model = PPO.load(ckpt, device="cpu")
    p = model.policy

    cfg.randomization.enabled = False
    env = StaggerEnv(cfg)
    env.set_level(level)
    xml_bytes = cfg.xml_path().read_bytes()

    contract = {
        "schema": SCHEMA,
        "meta": {
            "run": run.name,
            "checkpoint": ckpt.name,
            "timesteps": int(model.num_timesteps),
            "curriculum_level": level,
            "mujoco": mujoco.__version__,
            "xml_sha256": hashlib.sha256(xml_bytes).hexdigest(),
            "exported": dt.datetime.now().isoformat(timespec="seconds"),
        },
        "obs_dim": int(p.observation_space.shape[0]),
        "act_dim": int(p.action_space.shape[0]),
        "actor": export_mlp(p.mlp_extractor.policy_net, p.action_net),
        "critic": export_mlp(p.mlp_extractor.value_net, p.value_net),
        "value_range": value_range(model, env, args.value_episodes, seed=10_000),
        "obs_scales": cfg.obs.scales,
        "obs_clip": cfg.obs.clip,
        "default_joint_pos": [float(x) for x in env.robot.default_joint_pos],
        "action_scale": cfg.env.action_scale,
        "clip_actions": 1.0,
        "timestep": float(env.model.opt.timestep),
        "frame_skip": cfg.env.frame_skip,
        "home_yaw": 0.0,
        "standing_height": env.robot.standing_height,
        "falls": {
            "ground_bodies": cfg.falls.ground_bodies,
            "min_pelvis_height_frac": cfg.falls.min_pelvis_height_frac,
            "max_torso_tilt_deg": cfg.falls.max_torso_tilt_deg,
        },
        "hit_model": {
            # Strongest single hit the policy actually trained against (curriculum level reached),
            # vs the curriculum's ceiling. Weapons are tuned against j_max_trained.
            "j_max_trained": schedule_for_level(level, cfg.curriculum, list(REGIONS)).j_max,
            "j_max_curriculum": cfg.curriculum.j_max_final,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): run {run.name}, level {level}, value_range {contract['value_range']}")


if __name__ == "__main__":
    main()
