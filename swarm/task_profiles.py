"""Task profiles for Outcast Virus swarm gym scenarios.

Profiles keep scenario-specific reward weights, phases, and checkpoint metrics out
of the shared environment implementation. Runtime entity state still lives in
``swarm.env.SwarmEnv``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TASK_DIM = 16


@dataclass(frozen=True)
class TaskProfile:
    env_id: str
    primary_metric: str
    primary_mode: Literal["max", "min"]
    coverage_weight: float
    reward_weights: dict[str, float] = field(default_factory=dict)
    task_dim: int = TASK_DIM
    phase_names: tuple[str, ...] = ()
    metric_label: str = "Task score"


PROFILES: dict[str, TaskProfile] = {
    "drone-vs-drone": TaskProfile(
        env_id="drone-vs-drone",
        primary_metric="task_score",
        primary_mode="max",
        # Coverage is a competing objective for a pure combat task (it pays the
        # agent to patrol instead of chasing hostiles), so zero it out — the only
        # gradient should be engagement.
        coverage_weight=0.0,
        phase_names=("engage", "orbit"),
        metric_label="Kill + orbit score",
        reward_weights={
            "approach": 0.18,
            "closing": 0.22,
            # Potential-based Δdistance-to-nearest-hostile shaping (mirrors the
            # navigate-to-target progress term): a clean monotone gradient toward
            # contact. Fixes the flat "kind of near a hostile" plateau.
            "progress": 0.6,
            "kill": 6.0,
            "orbit": 0.35,
            "pressure": 0.45,
            "swarm": 0.35,
        },
    ),
    "moving-target-track": TaskProfile(
        env_id="moving-target-track",
        primary_metric="custody_fraction",
        primary_mode="max",
        coverage_weight=0.08,
        phase_names=("track",),
        metric_label="Custody %",
        reward_weights={
            "custody": 1.2,
            "angle_spread": 0.35,
            "distance": 0.2,
            "lost": 0.6,
        },
    ),
    "search-and-interdict": TaskProfile(
        env_id="search-and-interdict",
        primary_metric="task_score",
        primary_mode="max",
        coverage_weight=0.8,
        phase_names=("search", "contact", "intercept"),
        metric_label="Contact/intercept score",
        reward_weights={
            "contact": 2.0,
            "intercept": 3.0,
            "approach": 0.4,
            "delay": 0.15,
        },
    ),
    "defend-asset": TaskProfile(
        env_id="defend-asset",
        primary_metric="asset_integrity",
        primary_mode="max",
        coverage_weight=0.0,
        phase_names=("defend",),
        metric_label="Asset integrity",
        reward_weights={
            "intercept": 2.5,
            "ring": 0.45,
            "sector": 0.2,
            "breach": 4.0,
        },
    ),
    "hunt-and-seek": TaskProfile(
        env_id="hunt-and-seek",
        primary_metric="captures",
        primary_mode="max",
        coverage_weight=0.0,  # 3D hunt env owns its own search/coverage shaping
        phase_names=("search", "pursue", "capture"),
        metric_label="Captures",
        reward_weights={},  # reward shaping lives in swarm/hunt_env.py
    ),
    "navigate-to-target": TaskProfile(
        env_id="navigate-to-target",
        primary_metric="task_score",
        primary_mode="max",
        coverage_weight=0.0,
        phase_names=("navigate", "reached"),
        metric_label="Navigation score",
        reward_weights={
            "approach": 1.2,   # stronger goal pull (was 0.5)
            "reach": 8.0,      # bigger terminal bonus (was 5.0)
            "collision": 0.0,  # handled by env COLLISION_PENALTY
        },
    ),
}

DEFAULT_PROFILE = TaskProfile(
    env_id="default",
    primary_metric="coverage",
    primary_mode="max",
    coverage_weight=1.0,
    phase_names=("coverage",),
    reward_weights={},
)


def get_task_profile(env_id: str | None) -> TaskProfile:
    if env_id is None:
        return DEFAULT_PROFILE
    return PROFILES.get(env_id, DEFAULT_PROFILE)
