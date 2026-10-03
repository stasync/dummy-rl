"""Train the balance policy with PPO.

    python scripts/train.py --config configs/base.yaml --name A0_full
    python scripts/train.py --config configs/base.yaml --name smoke --timesteps 200000

Writes runs/<name>/: config.yaml (fully resolved), tb/ (TensorBoard), checkpoints/,
model.zip, summary.json (steps/s, final curriculum level, wall time).

Why PPO: it is the standard for simulated locomotion (legged_gym, MuJoCo Playground,
Isaac Lab all default to it). It is on-policy, so it pairs naturally with many cheap
parallel simulators, and its clipped update makes it forgiving of hyperparameters.
SAC is more sample-efficient but each sample costs far more compute, and with a fast
simulator samples are cheap -- wall-clock, not sample count, is what we're short of.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stagger.config import REPO_ROOT, Config, config_to_dict, load_config  # noqa: E402
from stagger.curriculum import CurriculumController  # noqa: E402
from stagger.env import make_env  # noqa: E402

ACTIVATIONS = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}


class StaggerCallback(BaseCallback):
    """Per-term reward logging + curriculum control.

    Every reward term is summed per episode and logged as rew/<term>, so TensorBoard shows
    what the return is made of, not just its total. Episode outcomes feed the curriculum.
    """

    def __init__(self, curriculum: CurriculumController, n_envs: int):
        super().__init__()
        self.curriculum = curriculum
        self.ep_terms = [defaultdict(float) for _ in range(n_envs)]
        self._reset_window()

    def _reset_window(self) -> None:
        self.win_terms: defaultdict[str, list[float]] = defaultdict(list)
        self.win_survived: list[bool] = []
        self.win_reasons: Counter[str] = Counter()
        self.win_hits = 0

    def _on_step(self) -> bool:
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            for k, v in info["reward_terms"].items():
                self.ep_terms[i][k] += v
            self.win_hits += info["hits"]
            if done:
                survived = info["knockdown"] == ""  # done without a knockdown = survived to the time limit
                for k, v in self.ep_terms[i].items():
                    self.win_terms[k].append(v)
                self.win_survived.append(survived)
                if not survived:
                    self.win_reasons[info["knockdown"]] += 1
                self.curriculum.record(info["level"], survived)
                self.ep_terms[i] = defaultdict(float)
        return True

    def _on_rollout_end(self) -> None:
        n = len(self.win_survived)
        if n:
            for k, v in self.win_terms.items():
                self.logger.record(f"rew/{k}", float(np.mean(v)))
            self.logger.record("ep/survival", float(np.mean(self.win_survived)))
            for reason, c in self.win_reasons.items():
                self.logger.record(f"ko/{reason}", c / n)
        self.logger.record("curriculum/level", self.curriculum.level)
        rate = self.curriculum.survival()
        if rate is not None:
            self.logger.record("curriculum/window_survival", rate)
        self.logger.record("ep/hits_per_rollout", self.win_hits)
        self._reset_window()

        if self.curriculum.update():
            self.training_env.env_method("set_level", self.curriculum.level)
            print(f"[curriculum] step {self.num_timesteps:,}: level -> {self.curriculum.level}", flush=True)


def build_model(cfg: Config, env, tb_dir: Path) -> PPO:
    p = cfg.ppo
    n_envs = env.num_envs
    batch_size = p.batch_size or (n_envs * p.n_steps) // 4
    lr = p.learning_rate
    learning_rate = (lambda progress_remaining: lr * progress_remaining) if p.lr_schedule == "linear" else lr
    policy_kwargs = dict(
        # dict(pi=..., vf=...) = two separate MLPs: the critic's value targets can't disturb the actor's features.
        net_arch=dict(pi=list(p.net_arch), vf=list(p.net_arch)),
        activation_fn=ACTIVATIONS[p.activation],
        # Initial action std = e^-1 ~ 0.37 (in units of action_scale): enough exploration
        # to discover stepping without flailing so hard it never sees a full episode.
        log_std_init=p.log_std_init,
    )
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=p.n_steps,
        batch_size=batch_size,
        n_epochs=p.n_epochs,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        clip_range=p.clip_range,
        ent_coef=p.ent_coef,
        vf_coef=p.vf_coef,
        max_grad_norm=p.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        seed=cfg.seed,
        device="cpu",
        verbose=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True, help="run name; output goes to runs/<name>/")
    ap.add_argument("--timesteps", type=int, default=None, help="override ppo.total_timesteps (quick runs)")
    ap.add_argument("--n-envs", type=int, default=None, help="override ppo.n_envs")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing runs/<name>/ first")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.timesteps:
        cfg.ppo.total_timesteps = args.timesteps
    if args.n_envs:
        cfg.ppo.n_envs = args.n_envs
    n_envs = cfg.ppo.n_envs or os.cpu_count() or 1
    torch.set_num_threads(cfg.ppo.torch_threads)

    run_dir = REPO_ROOT / "runs" / args.name
    if run_dir.exists():
        if not args.overwrite:
            sys.exit(f"{run_dir} exists; pick another --name or pass --overwrite")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        f"# resolved from {os.path.relpath(cfg.source, REPO_ROOT)}\n" + yaml.safe_dump(config_to_dict(cfg), sort_keys=False)
    )

    env = VecMonitor(SubprocVecEnv([make_env(cfg) for _ in range(n_envs)]))
    model = build_model(cfg, env, run_dir / "tb")
    curriculum = CurriculumController(cfg.curriculum)
    callbacks = CallbackList(
        [
            StaggerCallback(curriculum, n_envs),
            CheckpointCallback(
                save_freq=max(1, cfg.ppo.checkpoint_every_steps // n_envs),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo",
            ),
        ]
    )

    print(f"run {args.name}: {n_envs} envs, {cfg.ppo.total_timesteps:,} steps, batch {model.batch_size}", flush=True)
    t0 = time.time()
    try:
        model.learn(total_timesteps=cfg.ppo.total_timesteps, callback=callbacks, tb_log_name="ppo")
    finally:
        wall = time.time() - t0
        model.save(run_dir / "model.zip")
        summary = {
            "name": args.name,
            "config": os.path.relpath(cfg.source, REPO_ROOT),
            "timesteps": int(model.num_timesteps),
            "wall_s": round(wall, 1),
            "steps_per_s": round(model.num_timesteps / max(wall, 1e-9)),
            "n_envs": n_envs,
            "final_curriculum_level": curriculum.level,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        env.close()


if __name__ == "__main__":
    main()
