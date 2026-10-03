"""YAML config -> typed dataclasses.

A config file may start with `inherit: <path>` (relative to itself); the parent is loaded
first and this file's keys are deep-merged on top. That is how ablations stay tiny.
Unknown keys raise, so a typo in a YAML file fails loudly instead of being ignored.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EnvCfg:
    xml: str
    keyframe: str
    frame_skip: int
    episode_steps: int
    settle_steps: int
    action_scale: float
    reset_joint_noise: float
    reset_yaw_noise: float


@dataclass
class ObsCfg:
    scales: dict[str, float]
    clip: float


@dataclass
class FallsCfg:
    ground_bodies: list[str]
    min_pelvis_height_frac: float
    max_torso_tilt_deg: float


@dataclass
class HitsCfg:
    region_weights: dict[str, float]
    magnitude_min_frac: float
    max_elevation_deg: float
    interval_max_s: float


@dataclass
class CurriculumCfg:
    enabled: bool
    start_level: int
    max_level: int
    j_max_final: float
    core_regions: list[str]
    all_regions_from_level: int
    interval_min_start_s: float
    interval_min_final_s: float
    burst_from_level: int
    burst_prob: float
    burst_pellets: list[int]
    burst_total_scale: float
    promote_survival: float
    demote_survival: float
    window_episodes: int


@dataclass
class RandomizationCfg:
    enabled: bool
    friction: list[float]
    mass_scale: list[float]
    kp_scale: list[float]
    force_limit_scale: list[float]
    obs_noise_std: float
    action_latency_steps: list[int]


@dataclass
class RewardsCfg:
    weights: dict[str, float]
    params: dict[str, float]


@dataclass
class PPOCfg:
    total_timesteps: int
    n_envs: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    lr_schedule: str
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    net_arch: list[int]
    activation: str
    log_std_init: float
    torch_threads: int
    checkpoint_every_steps: int


@dataclass
class Config:
    seed: int
    env: EnvCfg
    obs: ObsCfg
    falls: FallsCfg
    hits: HitsCfg
    curriculum: CurriculumCfg
    randomization: RandomizationCfg
    rewards: RewardsCfg
    ppo: PPOCfg
    source: str = field(default="", repr=False)  # path the config was loaded from

    def xml_path(self) -> Path:
        p = Path(self.env.xml)
        return p if p.is_absolute() else REPO_ROOT / p


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_with_inheritance(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    return _deep_merge(_read_with_inheritance((path.parent / parent).resolve()), raw)


def _build(cls, data, where: str):
    """Recursively build dataclass `cls` from a dict, rejecting unknown/missing keys."""
    if not dataclasses.is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"{where}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for name, f in fields.items():
        if name not in data:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise KeyError(f"{where}: missing key '{name}'")
            continue
        t = hints[name]
        is_dc = isinstance(t, type) and dataclasses.is_dataclass(t)
        kwargs[name] = _build(t, data[name], f"{where}.{name}") if is_dc else data[name]
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    cfg = _build(Config, _read_with_inheritance(path), path.name)
    cfg.source = str(path)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    d = dataclasses.asdict(cfg)
    d.pop("source", None)
    return d

