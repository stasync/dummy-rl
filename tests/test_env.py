"""Env, obs, hits, falls, rewards, curriculum and config tests."""

import math

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from stagger.config import REPO_ROOT, load_config
from stagger.curriculum import CurriculumController
from stagger.env import StaggerEnv
from stagger.hits import Hit, apply_hits, clear_hits
from stagger.obs import OBS_DIM, build_obs
from stagger.rewards import TERMS, RewardFn
from stagger.robot import set_limp

BASE = REPO_ROOT / "configs/base.yaml"


@pytest.fixture()
def cfg():
    return load_config(BASE)


@pytest.fixture()
def env(cfg):
    return StaggerEnv(cfg)


def test_check_env(env):
    check_env(env, skip_render_check=True)


def test_obs_at_default_pose(env, cfg):
    d = env.data
    mujoco.mj_resetDataKeyframe(env.model, d, env.key_id)
    obs = build_obs(d.qpos, d.qvel, np.zeros(18), env.robot.default_joint_pos, d.qpos[:2], 0.0, cfg.obs.scales, cfg.obs.clip)
    assert obs.shape == (OBS_DIM,)
    np.testing.assert_allclose(obs[3:6], [0, 0, -1], atol=1e-6)   # gravity points down
    np.testing.assert_allclose(obs[11:13], [0, 1], atol=1e-6)    # no heading error
    np.testing.assert_allclose(obs[13:31], 0, atol=1e-6)         # at the default pose


def test_obs_body_frame_velocity(env, cfg):
    """Robot turned 90 deg left, moving +x in the world -> moving to its right (-y) in its own frame."""
    d = env.data
    mujoco.mj_resetDataKeyframe(env.model, d, env.key_id)
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    d.qvel[0:3] = [1.0, 0, 0]
    obs = build_obs(d.qpos, d.qvel, np.zeros(18), env.robot.default_joint_pos, d.qpos[:2], 0.0, cfg.obs.scales, cfg.obs.clip)
    np.testing.assert_allclose(obs[6:9], [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(obs[11:13], [1, 0], atol=1e-6)  # heading error +90 deg: sin=1, cos=0


@pytest.mark.parametrize("body", ["pelvis", "head", "lower_arm_L"])
def test_hit_delivers_its_impulse(env, body):
    """In free fall with no gravity and no contacts, total momentum change == J, wherever it hits.

    Uses RK4: the training integrator (implicitfast, first order) is off by 1-5% here because
    the limbs swing during the 20 ms push and a first-order step doesn't exactly conserve
    momentum. RK4 gives J to 1e-6, which isolates the hit-model math from integration error.
    """
    m, d = env.model, env.data
    set_limp(m, True)
    m.opt.gravity[:] = 0
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    mujoco.mj_resetDataKeyframe(m, d, env.key_id)
    d.qpos[2] = 3.0  # in the air
    mujoco.mj_forward(m, d)
    bid = env.robot.body_ids[body]
    J = np.array([12.0, -5.0, 2.0])
    point = d.xipos[bid] + np.array([0.02, 0.03, 0.04])  # off-center: adds spin, not momentum
    apply_hits(m, d, [Hit(bid, point, J)], env.frame_skip)
    mujoco.mj_step(m, d, nstep=env.frame_skip)
    clear_hits(d)
    mujoco.mj_forward(m, d)
    mujoco.mj_subtreeVel(m, d)  # total linear momentum = M * v_com of the root subtree
    momentum = m.body_subtreemass[1] * d.subtree_linvel[1]
    np.testing.assert_allclose(momentum, J, rtol=1e-6, atol=1e-6)
    assert not d.xfrc_applied.any()


def test_knockdown_when_lying(env):
    m, d = env.model, env.data
    mujoco.mj_resetDataKeyframe(m, d, env.key_id)
    d.qpos[2] = 0.15
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]  # pitched 90 deg (face down)
    mujoco.mj_forward(m, d)
    assert env.falls.check(d) != ""


def test_standing_is_not_knockdown(env):
    env.reset(seed=0)
    assert env.falls.check(env.data) == ""


def test_stiff_survives_level0(env):
    env.reset(seed=0)
    for _ in range(env.cfg.env.episode_steps):
        _, _, terminated, truncated, info = env.step(np.zeros(18))
        assert not terminated, info["knockdown"]
    assert truncated
    assert set(info["reward_terms"]) == set(TERMS)  # every term reported every step


def test_hits_happen_and_respect_schedule(env):
    env.set_level(5)
    env.reset(seed=1)
    n_hits = 0
    for _ in range(env.cfg.env.settle_steps):
        _, _, term, _, info = env.step(np.zeros(18))
        assert info["hits"] == 0  # no hits while settling
    for _ in range(200):
        _, _, term, _, info = env.step(np.zeros(18))
        n_hits += info["hits"]
        if term:
            break
    assert n_hits > 0


def test_unknown_reward_term_rejected(cfg):
    with pytest.raises(KeyError):
        RewardFn({**cfg.rewards.weights, "typo_term": 1.0}, cfg.rewards.params)


def test_unknown_config_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(f"inherit: {BASE}\nenv:\n  frame_skp: 5\n")
    with pytest.raises(KeyError):
        load_config(p)


def test_ablations_load():
    for p in (REPO_ROOT / "configs/ablations").glob("*.yaml"):
        c = load_config(p)
        assert c.ppo.total_timesteps == 10_000_000
    assert not load_config(REPO_ROOT / "configs/ablations/A1_no_curriculum.yaml").curriculum.enabled
    assert load_config(REPO_ROOT / "configs/ablations/A2_no_home.yaml").rewards.weights["home"] == 0.0


def test_curriculum_promote_and_demote(cfg):
    c = CurriculumController(cfg.curriculum)
    n = cfg.curriculum.window_episodes
    for _ in range(n):
        c.record(0, True)
    assert c.update() and c.level == 1
    for _ in range(n):
        c.record(0, True)  # episodes from the old level don't count
    assert not c.update()
    for _ in range(n):
        c.record(1, False)
    assert c.update() and c.level == 0


def test_point_segment_distance():
    from stagger.rewards import point_segment_distance

    a, b = np.array([0.0, -0.1]), np.array([0.0, 0.1])
    assert point_segment_distance(np.array([0.0, 0.0]), a, b) == pytest.approx(0.0)
    assert point_segment_distance(np.array([0.3, 0.05]), a, b) == pytest.approx(0.3)
    assert point_segment_distance(np.array([0.0, 0.4]), a, b) == pytest.approx(0.3)


def test_capture_term_full_when_standing_still(cfg):
    cfg.rewards.weights["capture"] = 1.0
    e = StaggerEnv(cfg)
    e.reset(seed=0)
    _, _, _, _, info = e.step(np.zeros(18))
    assert info["reward_terms"]["capture"] > 0.95


@pytest.mark.parametrize("pattern", ["single", "burst", "rapid"])
def test_hit_patterns(cfg, pattern):
    """Each event type delivers its hits: burst = n hits in one step, rapid = n hits spread out."""
    cfg.hits.patterns = {pattern: 1.0}
    cfg.curriculum.patterns_from_level = 0
    cfg.hits.pattern_hits = [4, 4]
    e = StaggerEnv(cfg)
    e.set_level(4)
    e.reset(seed=0)
    hits_per_step = []
    while sum(hits_per_step) == 0 or len(hits_per_step) < e.cfg.env.settle_steps + 200:
        _, _, term, _, info = e.step(np.zeros(18))
        hits_per_step.append(info["hits"])
        if term:
            break
    first = next(i for i, h in enumerate(hits_per_step) if h)
    if pattern == "single":
        assert hits_per_step[first] == 1
    elif pattern == "burst":
        assert hits_per_step[first] == 4
    else:
        spacing = round(cfg.hits.rapid_interval_s / e.dt)
        assert [hits_per_step[first + i * spacing] for i in range(4)] == [1, 1, 1, 1]


def test_step_counter(env):
    """A foot that lifts off and lands 10 cm away is one step; landing in place is not."""
    env.reset(seed=0)
    xy = env._foot_xy()
    assert env._count_steps(np.array([False, True]), xy) == 0          # left foot lifts
    moved = xy.copy()
    moved[0] += [0.1, 0.0]
    assert env._count_steps(np.array([True, True]), moved) == 1        # lands 10 cm away
    assert env._count_steps(np.array([True, False]), moved) == 0       # right lifts
    assert env._count_steps(np.array([True, True]), moved) == 0        # lands in place


def test_old_run_configs_still_load():
    """Configs saved by Day 1 runs used curriculum.burst_*; they must still load for eval."""
    p = REPO_ROOT / "runs/D1_short/config.yaml"
    if not p.exists():
        pytest.skip("no Day 1 run on this machine")
    c = load_config(p)
    assert c.hits.patterns["burst"] == pytest.approx(0.15)
