"""Model loading, joint/body indexing, default pose and hit regions.

Everything else (env, eval, export) asks this module for ids instead of hardcoding them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# Hit regions (PLAN.md §4.4). Body names come from assets/stagger.xml.
REGIONS: dict[str, list[str]] = {
    "torso": ["torso"],
    "pelvis": ["pelvis"],
    "head": ["head"],
    "arms": ["upper_arm_L", "upper_arm_R", "lower_arm_L", "lower_arm_R"],
    "legs": ["thigh_L", "thigh_R", "shin_L", "shin_R", "foot_L", "foot_R"],
}

FOOT_BODIES = ("foot_L", "foot_R")

_ACTUATION_BIT = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)


@dataclass
class Robot:
    model: mujoco.MjModel
    default_qpos: np.ndarray        # full qpos of the "stand" keyframe (nq,)
    default_joint_pos: np.ndarray   # actuated joints only (nu,)
    standing_height: float          # pelvis z in the default pose (h0)
    joint_lo: np.ndarray            # joint range per actuated joint (nu,)
    joint_hi: np.ndarray
    body_ids: dict[str, int]
    region_bodies: dict[str, list[int]]
    floor_geom: int
    foot_geoms: list[int]

    @property
    def nu(self) -> int:
        return self.model.nu

    def body_main_geom(self, body_id: int) -> int:
        """Geom used for sampling hit points: by convention it has the same name as its body."""
        return self.model.geom(self.model.body(body_id).name).id

    def geoms_of_bodies(self, names: list[str]) -> list[int]:
        ids = {self.body_ids[n] for n in names}
        return [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in ids]


def load_robot(xml_path: str | Path, keyframe: str = "stand") -> Robot:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    key = model.key(keyframe)
    default_qpos = key.qpos.copy()

    # The obs/action code assumes actuator i drives the joint whose qpos lives at 7 + i
    # (right after the 7-dim freejoint). Check it once here instead of trusting the XML order.
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        assert model.jnt_qposadr[jid] == 7 + i, f"actuator {model.actuator(i).name} out of joint order"
        assert model.jnt_dofadr[jid] == 6 + i
    joint_ids = model.actuator_trnid[:, 0]

    body_ids = {model.body(b).name: b for b in range(1, model.nbody)}
    return Robot(
        model=model,
        default_qpos=default_qpos,
        default_joint_pos=default_qpos[7:].copy(),
        standing_height=float(default_qpos[2]),
        joint_lo=model.jnt_range[joint_ids, 0].copy(),
        joint_hi=model.jnt_range[joint_ids, 1].copy(),
        body_ids=body_ids,
        region_bodies={r: [body_ids[n] for n in names] for r, names in REGIONS.items()},
        floor_geom=model.geom("floor").id,
        foot_geoms=[model.geom(n).id for n in FOOT_BODIES],
    )


def set_limp(model: mujoco.MjModel, limp: bool) -> None:
    """Limp mode: switch all motors off (the robot collapses like a puppet with cut strings)."""
    if limp:
        model.opt.disableflags |= _ACTUATION_BIT
    else:
        model.opt.disableflags &= ~_ACTUATION_BIT
