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

STEP_MIN_DIST = 0.05  # m: a foot that lifts off and lands this far away counts as a recovery step


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
        # Linear inverted pendulum: a CoM at height h moving at v comes to rest over a point
        # v / omega0 ahead of it (the capture point), omega0 = sqrt(g / h).
        self.omega0 = math.sqrt(-self.model.opt.gravity[2] / self.robot.standing_com_height)

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
        self.hit_queue: list[tuple[int, int, float, str]] = []  # (due step, n hits, impulse per hit, region)
        self.foot_down = np.ones(2, dtype=bool)
        self.liftoff_xy = np.zeros((2, 2))

    # --- curriculum hook (called from the training callback via env_method) ----------------

    def set_level(self, level: int) -> None:
        self.level = int(level)
        self.schedule = schedule_for_level(self.level, self.cfg.curriculum, list(REGIONS), self.cfg.hits.patterns)

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
        self.hit_queue = []
        self.foot_down[:] = True
        self.liftoff_xy[:] = self._foot_xy()
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

        new_event = self.schedule.j_max > 0.0 and self.step_count >= self.next_hit_step
        if new_event:
            span = self._queue_hit_event()
            self.next_hit_step = self.step_count + span + self._sample_interval_steps()
        self.last_hits = self._due_hits()
        if self.last_hits:
            apply_hits(m, d, self.last_hits, self.frame_skip)

        mujoco.mj_step(m, d, nstep=self.frame_skip)
        if self.last_hits:
            clear_hits(d)
        self.step_count += 1

        reason = self.falls.check(d)
        knocked_down = reason != ""
        foot_contact, foot_xy = self._foot_contacts(), self._foot_xy()
        steps_taken = self._count_steps(foot_contact, foot_xy)
        reward, terms = self.reward_fn(self._reward_state(a, knocked_down, foot_contact, foot_xy))
        self.prev_action = a

        terminated = knocked_down
        truncated = not knocked_down and self.step_count >= self.cfg.env.episode_steps
        info = {
            "reward_terms": terms,
            "knockdown": reason,
            "level": self.episode_level,
            "hits": len(self.last_hits),
            "hit_events": int(new_event),
            "steps_taken": steps_taken,
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

    @staticmethod
    def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
        keys = list(weights)
        p = np.array([weights[k] for k in keys], dtype=float)
        return keys[rng.choice(len(keys), p=p / p.sum())]

    def _queue_hit_event(self) -> int:
        """Schedule one hit event, like a weapon would deliver it. Returns how many steps it spans.

        single: one hit.  burst (shotgun): n pellets in the same step.  rapid (rifle): n hits
        rapid_interval_s apart. All hits of an event land in one region (the player aims there).
        A multi-hit event carries pattern_total_scale x the impulse of a single hit, split evenly.
        """
        rng, h, sched = self.np_random, self.cfg.hits, self.schedule
        region = self._pick(rng, {r: w for r, w in h.region_weights.items() if r in sched.regions})
        pattern = self._pick(rng, sched.patterns)
        total = rng.uniform(h.magnitude_min_frac, 1.0) * sched.j_max
        now = self.step_count
        if pattern == "single":
            self.hit_queue.append((now, 1, total, region))
            return 0
        n = int(rng.integers(h.pattern_hits[0], h.pattern_hits[1] + 1))
        per_hit = total * h.pattern_total_scale / n
        if pattern == "burst":
            self.hit_queue.append((now, n, per_hit, region))
            return 0
        spacing = max(1, round(h.rapid_interval_s / self.dt))
        self.hit_queue += [(now + i * spacing, 1, per_hit, region) for i in range(n)]
        return (n - 1) * spacing

    def _due_hits(self) -> list[Hit]:
        """Hits whose time has come. Points are sampled now, on the body's current pose."""
        due = [q for q in self.hit_queue if q[0] <= self.step_count]
        if not due:
            return []
        self.hit_queue = [q for q in self.hit_queue if q[0] > self.step_count]
        rng, elev = self.np_random, self.cfg.hits.max_elevation_deg
        return [
            sample_hit(rng, self.robot, self.data, {region: 1.0}, per_hit, elev)
            for _, n, per_hit, region in due
            for _ in range(n)
        ]

    def _foot_contacts(self) -> np.ndarray:
        d, r = self.data, self.robot
        contact = np.zeros(2, dtype=bool)
        for i in range(d.ncon):
            g1, g2 = d.contact.geom[i]
            for k, fg in enumerate(r.foot_geoms):
                if (g1 == fg and g2 == r.floor_geom) or (g2 == fg and g1 == r.floor_geom):
                    contact[k] = True
        return contact

    def _foot_xy(self) -> np.ndarray:
        return self.data.geom_xpos[self.robot.foot_geoms, :2].copy()

    def _count_steps(self, contact: np.ndarray, foot_xy: np.ndarray) -> int:
        """Recovery steps: a foot that lifts off and touches down >= STEP_MIN_DIST from where it lifted."""
        steps = 0
        for k in range(2):
            if self.foot_down[k] and not contact[k]:
                self.liftoff_xy[k] = foot_xy[k]
            elif not self.foot_down[k] and contact[k]:
                steps += int(np.linalg.norm(foot_xy[k] - self.liftoff_xy[k]) >= STEP_MIN_DIST)
        self.foot_down[:] = contact
        return steps

    def _reward_state(self, action: np.ndarray, knocked_down: bool, foot_contact: np.ndarray, foot_xy: np.ndarray) -> RewardState:
        m, d, r = self.model, self.data, self.robot
        R = quat_to_mat(d.qpos[3:7])

        foot_vel = np.zeros((2, 2))
        vel6 = np.zeros(6)
        for k, b in enumerate(self.foot_body_ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)  # [ang, lin], world frame
            foot_vel[k] = vel6[3:5]

        mujoco.mj_subtreeVel(m, d)  # fills subtree_linvel (CoM velocity of each subtree)
        root = r.body_ids["pelvis"]  # pelvis subtree = the whole robot
        capture_point = d.subtree_com[root, :2] + d.subtree_linvel[root, :2] / self.omega0

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
            foot_xy=foot_xy,
            capture_point=capture_point,
            knocked_down=knocked_down,
        )


def make_env(cfg: Config):
    """Factory for SubprocVecEnv: each worker builds its own env (and its own MjModel).
    Seeding happens through the VecEnv (PPO(seed=...) seeds worker i with seed + i)."""

    def _init():
        return StaggerEnv(cfg)

    return _init
