"""Reward-term registry (PLAN.md §4.6).

Every term is a small named function of a RewardState. The YAML config supplies one weight
per term; reward = sum(weight * term). Each weighted term is also returned separately so
training can log it, which is how you find out *why* a policy behaves the way it does
(e.g. "posture is eating 40% of the return, so it refuses to step").

Positive terms use exp(-err^2 / sigma) "kernels": bounded in (0, 1], 1 at perfect, smooth
everywhere. Unlike a raw -err^2 penalty, one huge error after a big hit can't swamp the
whole episode's return, and there is still a gradient toward "less wrong".
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class RewardState:
    """Everything a reward term may look at, computed once per control step by the env."""

    gravity_body: np.ndarray     # (3,) gravity direction in pelvis frame; xy = 0 when upright
    pelvis_height: float
    standing_height: float
    base_xy: np.ndarray          # (2,)
    home_xy: np.ndarray          # (2,)
    yaw_error: float             # rad, wrapped to [-pi, pi]
    base_lin_vel: np.ndarray     # (3,) world frame
    joint_pos: np.ndarray        # (nu,)
    default_joint_pos: np.ndarray
    joint_vel: np.ndarray        # (nu,)
    joint_lo: np.ndarray
    joint_hi: np.ndarray
    actuator_force: np.ndarray   # (nu,) N*m
    action: np.ndarray           # (nu,) this step, in [-1, 1]
    prev_action: np.ndarray
    foot_in_contact: np.ndarray  # (2,) bool
    foot_vel_xy: np.ndarray      # (2, 2) world-frame xy velocity of each foot
    foot_xy: np.ndarray          # (2, 2) world xy of each foot's center
    capture_point: np.ndarray    # (2,) where the CoM would come to rest: com_xy + v_com_xy / omega0
    knocked_down: bool


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to the segment a-b (2D)."""
    ab, ap = b - a, p - a
    t = float(np.clip(ap @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(ap - t * ab))


TermFn = Callable[[RewardState, dict[str, float]], float]
TERMS: dict[str, TermFn] = {}


def term(name: str):
    def register(fn: TermFn) -> TermFn:
        TERMS[name] = fn
        return fn

    return register


@term("alive")
def alive(s: RewardState, p: dict[str, float]) -> float:
    # Constant bonus per step survived: the core "don't fall" signal.
    return 1.0


@term("upright")
def upright(s: RewardState, p: dict[str, float]) -> float:
    return math.exp(-float(s.gravity_body[0] ** 2 + s.gravity_body[1] ** 2) / p["upright_sigma"])


@term("height")
def height(s: RewardState, p: dict[str, float]) -> float:
    return math.exp(-((s.pelvis_height - s.standing_height) ** 2) / p["height_sigma"])


@term("home")
def home(s: RewardState, p: dict[str, float]) -> float:
    # Pulls it back to where it stood, so a recovery step is followed by stepping back.
    d = s.base_xy - s.home_xy
    return math.exp(-float(d @ d) / p["home_sigma"])


@term("heading")
def heading(s: RewardState, p: dict[str, float]) -> float:
    return math.exp(-(s.yaw_error**2) / p["heading_sigma"])


@term("calm")
def calm(s: RewardState, p: dict[str, float]) -> float:
    # Rewards standing still. Too much of this = stiff "statue" that topples instead of stepping.
    return math.exp(-float(s.base_lin_vel @ s.base_lin_vel) / p["calm_sigma"])


@term("posture")
def posture(s: RewardState, p: dict[str, float]) -> float:
    d = s.joint_pos - s.default_joint_pos
    return float(d @ d)


@term("energy")
def energy(s: RewardState, p: dict[str, float]) -> float:
    # Mechanical power |tau * qdot| summed over motors: discourages flailing and buzzing.
    return float(np.abs(s.actuator_force * s.joint_vel).sum())


@term("action_rate")
def action_rate(s: RewardState, p: dict[str, float]) -> float:
    # Smoothness: big jumps between consecutive targets look jittery and transfer badly.
    d = s.action - s.prev_action
    return float(d @ d)


@term("joint_limit")
def joint_limit(s: RewardState, p: dict[str, float]) -> float:
    mid = 0.5 * (s.joint_lo + s.joint_hi)
    half = 0.5 * (s.joint_hi - s.joint_lo) * p["joint_limit_soft_frac"]
    over = np.abs(s.joint_pos - mid) - half
    return float(np.clip(over, 0.0, None).sum())


@term("foot_slip")
def foot_slip(s: RewardState, p: dict[str, float]) -> float:
    # A planted foot should not skate; sliding feet are a common sim exploit.
    v2 = (s.foot_vel_xy**2).sum(axis=1)
    return float((v2 * s.foot_in_contact).sum())


@term("capture")
def capture(s: RewardState, p: dict[str, float]) -> float:
    # Capture point (CP): where the robot would have to put a foot to stop. Standing still, it is
    # under the CoM, between the feet -> 1. A hard push throws it outside the feet; the robot can
    # only bring this back up by moving a foot toward the CP (a recovery step) or by braking.
    # Unlike "alive", this pays off *during* the step, so PPO gets a gradient toward stepping
    # instead of having to discover a complete, well-timed step by chance.
    d = point_segment_distance(s.capture_point, s.foot_xy[0], s.foot_xy[1]) - p["capture_margin"]
    return math.exp(-max(0.0, d) ** 2 / p["capture_sigma"])


@term("termination")
def termination(s: RewardState, p: dict[str, float]) -> float:
    return 1.0 if s.knocked_down else 0.0


class RewardFn:
    def __init__(self, weights: dict[str, float], params: dict[str, float]):
        unknown = set(weights) - set(TERMS)
        if unknown:
            raise KeyError(f"unknown reward term(s) in config: {sorted(unknown)}; known: {sorted(TERMS)}")
        self.weights = {name: float(weights.get(name, 0.0)) for name in TERMS}
        self.params = params

    def __call__(self, s: RewardState) -> tuple[float, dict[str, float]]:
        terms = {name: w * TERMS[name](s, self.params) if w != 0.0 else 0.0 for name, w in self.weights.items()}
        return sum(terms.values()), terms
