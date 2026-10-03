"""StaggerEnv: stand on the spot while being shot at (PLAN.md §4).

One step = one 50 Hz policy decision = frame_skip (10) MuJoCo steps of 2 ms.
Action: 18 joint-target offsets in [-1, 1] -> PD target = default pose + action_scale * a.
The motors are PD position servos (in the XML); the policy only chooses where they aim.
This "residual around a default pose" action space is much easier to learn than raw torques,
and action = 0 is exactly Stiff mode, so the policy starts from a robot that already stands.
"""

from __future__ import annotations

import math

import gymnasium as gym
import mujoco
import numpy as np

from stagger.config import Config
from stagger.curriculum import schedule_for_level
from stagger.falls import FallDetector
from stagger.hits import Hit, apply_hits, clear_hits, sample_hit
from stagger.obs import OBS_DIM, build_obs, quat_to_mat, wrap_angle, yaw_from_mat
from stagger.randomization import Randomizer
from stagger.rewards import RewardFn, RewardState
from stagger.robot import REGIONS, load_robot


class StaggerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.robot = load_robot(cfg.xml_path(), cfg.env.keyframe)
        self.model = self.robot.model
        self.data = mujoco.MjData(self.model)
        self.key_id = self.model.key(cfg.env.keyframe).id
        self.frame_skip = cfg.env.frame_skip
        self.dt = self.frame_skip * self.model.opt.timestep

        self.falls = FallDetector(self.robot, cfg.falls)
        self.reward_fn = RewardFn(cfg.rewards.weights, cfg.rewards.params)
        self.randomizer = Randomizer(self.model, cfg.randomization)
        self.foot_body_ids = [self.model.geom_bodyid[g] for g in self.robot.foot_geoms]

        nu = self.robot.nu
        self.action_space = gym.spaces.Box(-1.0, 1.0, (nu,), np.float32)
        clip = cfg.obs.clip
        self.observation_space = gym.spaces.Box(-clip, clip, (OBS_DIM,), np.float32)

        self.set_level(cfg.curriculum.start_level if cfg.curriculum.enabled else cfg.curriculum.max_level)

        # Episode state (filled by reset)
        self.home_xy = np.zeros(2)
        self.home_yaw = 0.0
        self.prev_action = np.zeros(nu)
        self.pending_action = np.zeros(nu)
        self.step_count = 0
        self.next_hit_step = 0
        self.episode_level = self.level
        self.last_hits: list[Hit] = []

    # --- curriculum hook (called from the training callback via env_method) ----------------

    def set_level(self, level: int) -> None:
        self.level = int(level)
        self.schedule = schedule_for_level(self.level, self.cfg.curriculum, list(REGIONS))

    # --- gym API ----------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        m, d, e = self.model, self.data, self.cfg.env

        self.randomizer.reset(rng)
        mujoco.mj_resetDataKeyframe(m, d, self.key_id)
        d.qpos[7:] += rng.uniform(-e.reset_joint_noise, e.reset_joint_noise, self.robot.nu)
        yaw = rng.uniform(-e.reset_yaw_noise, e.reset_yaw_noise)
        d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.ctrl[:] = self.robot.default_joint_pos
        mujoco.mj_forward(m, d)

        # Home = where it spawned, facing the range (+x). The yaw noise above is an error to fix.
        self.home_xy = d.qpos[0:2].copy()
        self.home_yaw = 0.0
        self.prev_action[:] = 0.0
        self.pending_action[:] = 0.0
        self.step_count = 0
        self.next_hit_step = e.settle_steps + self._sample_interval_steps()
        self.episode_level = self.level
        self.last_hits = []
        return self._obs(), {}

    def step(self, action: np.ndarray):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Action latency (randomization): with 1 step of latency the motors get last step's action.
        if self.randomizer.latency_steps > 0:
            applied, self.pending_action = self.pending_action, a.copy()
        else:
            applied = a
        d.ctrl[:] = self.robot.default_joint_pos + self.cfg.env.action_scale * applied

        self.last_hits = []
        if self.schedule.j_max > 0.0 and self.step_count >= self.next_hit_step:
            self.last_hits = self._sample_hit_event()
            apply_hits(m, d, self.last_hits, self.frame_skip)
            self.next_hit_step = self.step_count + self._sample_interval_steps()

        mujoco.mj_step(m, d, nstep=self.frame_skip)
        if self.last_hits:
            clear_hits(d)
        self.step_count += 1

        reason = self.falls.check(d)
        knocked_down = reason != ""
        reward, terms = self.reward_fn(self._reward_state(a, knocked_down))
        self.prev_action = a

        terminated = knocked_down
        truncated = not knocked_down and self.step_count >= self.cfg.env.episode_steps
        info = {
            "reward_terms": terms,
            "knockdown": reason,
            "level": self.episode_level,
            "hits": len(self.last_hits),
        }
        return self._obs(), float(reward), terminated, truncated, info

    # --- internals --------------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        c = self.cfg.obs
        obs = build_obs(
            self.data.qpos, self.data.qvel, self.prev_action, self.robot.default_joint_pos,
            self.home_xy, self.home_yaw, c.scales, c.clip,
        )
        if self.randomizer.obs_noise_std > 0.0:
            obs = obs + self.np_random.normal(0.0, self.randomizer.obs_noise_std, obs.shape)
            obs = np.clip(obs, -c.clip, c.clip).astype(np.float32)
        return obs

    def _sample_interval_steps(self) -> int:
        s = self.np_random.uniform(self.schedule.interval_min_s, self.cfg.hits.interval_max_s)
        return max(1, round(s / self.dt))

    def _sample_hit_event(self) -> list[Hit]:
        """One normal hit, or (from burst_from_level) sometimes a shotgun-like burst."""
        rng, h, sched = self.np_random, self.cfg.hits, self.schedule
        weights = {r: w for r, w in h.region_weights.items() if r in sched.regions}
        magnitude = rng.uniform(h.magnitude_min_frac, 1.0) * sched.j_max
        if rng.random() < sched.burst_prob:
            lo, hi = self.cfg.curriculum.burst_pellets
            k = int(rng.integers(lo, hi + 1))
            regions = list(weights)
            p = np.array([weights[r] for r in regions])
            region = regions[rng.choice(len(regions), p=p / p.sum())]  # a blast lands in one area
            per_pellet = magnitude * self.cfg.curriculum.burst_total_scale / k
            return [sample_hit(rng, self.robot, self.data, {region: 1.0}, per_pellet, h.max_elevation_deg) for _ in range(k)]
        return [sample_hit(rng, self.robot, self.data, weights, magnitude, h.max_elevation_deg)]

    def _reward_state(self, action: np.ndarray, knocked_down: bool) -> RewardState:
        m, d, r = self.model, self.data, self.robot
        R = quat_to_mat(d.qpos[3:7])

        foot_contact = np.zeros(2, dtype=bool)
        for i in range(d.ncon):
            g1, g2 = d.contact.geom[i]
            for k, fg in enumerate(r.foot_geoms):
                if (g1 == fg and g2 == r.floor_geom) or (g2 == fg and g1 == r.floor_geom):
                    foot_contact[k] = True
        foot_vel = np.zeros((2, 2))
        vel6 = np.zeros(6)
        for k, b in enumerate(self.foot_body_ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)  # [ang, lin], world frame
            foot_vel[k] = vel6[3:5]

        return RewardState(
            gravity_body=-R[2, :],
            pelvis_height=float(d.qpos[2]),
            standing_height=r.standing_height,
            base_xy=d.qpos[0:2].copy(),
            home_xy=self.home_xy,
            yaw_error=wrap_angle(yaw_from_mat(R) - self.home_yaw),
            base_lin_vel=d.qvel[0:3].copy(),
            joint_pos=d.qpos[7:].copy(),
            default_joint_pos=r.default_joint_pos,
            joint_vel=d.qvel[6:].copy(),
            joint_lo=r.joint_lo,
            joint_hi=r.joint_hi,
            actuator_force=d.actuator_force.copy(),
            action=action,
            prev_action=self.prev_action,
            foot_in_contact=foot_contact,
            foot_vel_xy=foot_vel,
            knocked_down=knocked_down,
        )


def make_env(cfg: Config):
    """Factory for SubprocVecEnv: each worker builds its own env (and its own MjModel).
    Seeding happens through the VecEnv (PPO(seed=...) seeds worker i with seed + i)."""

    def _init():
        return StaggerEnv(cfg)

    return _init
