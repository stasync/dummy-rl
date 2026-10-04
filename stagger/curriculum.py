"""Hit curriculum: strength, location and frequency, driven by one global level (PLAN.md §4.7).

Why a curriculum: at full strength from step 0, an untrained policy falls on almost every
episode, so it mostly learns "falling happens" and gets little signal about *how* to stay up.
Starting with no hits and raising difficulty only once it survives keeps the task at the edge
of what it can do. Ablation A1 trains at max level from the start to show the difference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from stagger.config import CurriculumCfg


@dataclass
class HitSchedule:
    j_max: float                      # N*s, upper bound on one hit event's impulse
    regions: list[str]                # which hit regions are allowed
    interval_min_s: float             # time between hit events ~ U(interval_min_s, interval_max_s)
    patterns: dict[str, float]        # allowed event types and their weights


def schedule_for_level(level: int, cfg: CurriculumCfg, all_regions: list[str], patterns: dict[str, float]) -> HitSchedule:
    t = level / cfg.max_level  # 0..1
    return HitSchedule(
        j_max=cfg.j_max_final * t,
        regions=list(all_regions) if level >= cfg.all_regions_from_level else list(cfg.core_regions),
        interval_min_s=cfg.interval_min_start_s + t * (cfg.interval_min_final_s - cfg.interval_min_start_s),
        patterns=dict(patterns) if level >= cfg.patterns_from_level else {"single": 1.0},
    )


class CurriculumController:
    """Promote when survival at the current level > promote_survival, demote when < demote_survival.

    Only episodes that *started* at the current level count, so a level change never gets
    judged on episodes played under the previous difficulty.
    """

    def __init__(self, cfg: CurriculumCfg):
        self.cfg = cfg
        self.level = cfg.start_level if cfg.enabled else cfg.max_level
        self.results: deque[bool] = deque(maxlen=cfg.window_episodes)

    def record(self, episode_level: int, survived: bool) -> None:
        if episode_level == self.level:
            self.results.append(survived)

    def survival(self) -> float | None:
        if len(self.results) < self.cfg.window_episodes:
            return None
        return sum(self.results) / len(self.results)

    def update(self) -> bool:
        """Maybe change level. Returns True if it changed."""
        if not self.cfg.enabled:
            return False
        rate = self.survival()
        if rate is None:
            return False
        new = self.level
        if rate > self.cfg.promote_survival and self.level < self.cfg.max_level:
            new = self.level + 1
        elif rate < self.cfg.demote_survival and self.level > 0:
            new = self.level - 1
        if new == self.level:
            return False
        self.level = new
        self.results.clear()
        return True
