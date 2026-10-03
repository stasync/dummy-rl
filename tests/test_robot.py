"""Robot model acceptance (PLAN.md §4.1): Stiff stands, Limp collapses, indexing is sane."""

import mujoco
import numpy as np
import pytest

from stagger.config import REPO_ROOT
from stagger.robot import REGIONS, load_robot, set_limp

XML = REPO_ROOT / "assets/stagger.xml"


@pytest.fixture()
def robot():
    return load_robot(XML)


def _run(robot, seconds: float, limp: bool, gain_scale: float = 1.0):
    m = robot.model
    m.actuator_gainprm[:, 0] *= gain_scale
    m.actuator_biasprm[:, 1] *= gain_scale
    set_limp(m, limp)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_step(m, d, nstep=int(seconds / m.opt.timestep))
    return d


def test_size_and_mass(robot):
    m = robot.model
    assert (m.nq, m.nv, m.nu) == (25, 24, 18)
    assert 28.0 < m.body_subtreemass[1] < 32.0  # ~30 kg


def test_regions_cover_all_bodies(robot):
    named = {b for bodies in REGIONS.values() for b in bodies}
    assert named == set(robot.body_ids)


@pytest.mark.parametrize("gain_scale", [1.0, 0.9])  # 0.9 = weakest motors under randomization
def test_stiff_stands_10s(robot, gain_scale):
    d = _run(robot, 10.0, limp=False, gain_scale=gain_scale)
    up = d.xmat[robot.body_ids["torso"]][8]
    assert up > 0.99
    assert d.qpos[2] > 0.95 * robot.standing_height


def test_limp_collapses(robot):
    d = _run(robot, 3.0, limp=True)
    assert d.qpos[2] < 0.6 * robot.standing_height
