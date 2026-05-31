"""Task-aligned policy evaluation for CombatOS swarm scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from swarm.models import Actor
from swarm.scenarios import make_scenario_env
from swarm.task_profiles import get_task_profile


@dataclass(frozen=True)
class EvalResult:
    primary_metric: str
    primary_value: float
    task_score: float
    coverage: float
    metrics: dict[str, float]


def _numeric_metrics(info: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in info.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
    task_metrics = info.get("task_metrics")
    if isinstance(task_metrics, dict):
        for key, value in task_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                out[key] = float(value)
    return out


def eval_policy(
    actor: Actor | None,
    *,
    env_id: str,
    battlefield,
    n_episodes: int = 5,
    seed: int = 1234,
    deterministic: bool = True,
) -> EvalResult:
    """Run deterministic evaluation and aggregate task KPIs.

    ``actor=None`` evaluates a random policy baseline.
    """
    profile = get_task_profile(env_id)
    episode_metrics: list[dict[str, float]] = []

    for ep in range(n_episodes):
        env = make_scenario_env(env_id, battlefield=battlefield, seed=seed + ep)
        obs = env.reset(seed=seed + ep)
        done = False
        info: dict[str, Any] = {"coverage": 0.0, "task_score": 0.0}
        while not done:
            if actor is None:
                action = env.rng.uniform(-1, 1, size=(env.n, env.act_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs)
                    if deterministic:
                        action = actor(obs_t).numpy().astype(np.float32)
                    else:
                        sampled, _ = actor.sample(obs_t)
                        action = sampled.numpy().astype(np.float32)
            obs, _, dones, info = env.step(action)
            done = bool(dones.any())
        episode_metrics.append(_numeric_metrics(info))

    keys = sorted({key for metrics in episode_metrics for key in metrics})
    metrics = {
        key: float(np.mean([m.get(key, 0.0) for m in episode_metrics]))
        for key in keys
    }
    coverage = metrics.get("coverage", 0.0)
    task_score = metrics.get("task_score", coverage)
    primary_value = metrics.get(profile.primary_metric, metrics.get("primary_value", task_score))
    return EvalResult(
        primary_metric=profile.primary_metric,
        primary_value=float(primary_value),
        task_score=float(task_score),
        coverage=float(coverage),
        metrics=metrics,
    )


def is_better(candidate: EvalResult, incumbent: EvalResult | None, env_id: str) -> bool:
    if incumbent is None:
        return True
    profile = get_task_profile(env_id)
    if profile.primary_mode == "min":
        if candidate.primary_value != incumbent.primary_value:
            return candidate.primary_value < incumbent.primary_value
    else:
        if candidate.primary_value != incumbent.primary_value:
            return candidate.primary_value > incumbent.primary_value
    if candidate.task_score != incumbent.task_score:
        return candidate.task_score > incumbent.task_score
    return candidate.coverage > incumbent.coverage
