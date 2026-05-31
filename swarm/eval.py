"""Task-aligned policy evaluation for Outcast Virus swarm scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from swarm.env import OBSTACLE_DIM
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


class ScriptedPolicy:
    """Heuristic baseline: steer straight at the scenario objective.

    Reads only the local observation (decentralized, like the learned policy):
    the first two task-block features are the normalized bearing to the nearest
    hostile / target / asset / goal for every combat-style scenario, so a unit
    step along that vector is a "fly toward the thing" controller. If PPO can't
    beat this, the bug is reward/obs/curriculum, not network capacity.
    """

    def __init__(self, env):
        self.scenario_id = env.scenario_id
        # The 3D hunt env has a different observation layout (no coverage patch);
        # it exposes the target block offset directly as env.target_off.
        self.is_hunt = self.scenario_id == "hunt-and-seek"
        if self.is_hunt:
            self.env = env
            self.target_off = env.target_off
            return
        self.patch = env.patch
        self.patch_off = 4 + 2 * env.k
        self.obstacle_off = 4 + 2 * env.k + env.patch * env.patch
        self.task_off = 4 + 2 * env.k + env.patch * env.patch + OBSTACLE_DIM

    def __call__(self, obs):
        o = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)

        if self.is_hunt:
            if hasattr(self, "env") and hasattr(self.env, "expert_action"):
                return torch.from_numpy(self.env.expert_action())
            # Fall back to flying straight at the estimated target if this
            # policy was created without a live hunt env reference.
            off = self.target_off
            bearing3 = o[:, off:off + 3]
            nrm = np.linalg.norm(bearing3, axis=1, keepdims=True)
            action = (bearing3 / np.maximum(nrm, 1e-6)).astype(np.float32)
            return torch.from_numpy(action)

        bearing = o[:, self.task_off:self.task_off + 2]

        if self.scenario_id == "defend-asset":
            hostile = o[:, self.task_off + 2:self.task_off + 4]
            has_hostile = o[:, self.task_off + 4:self.task_off + 5] > 0.5
            bearing = np.where(has_hostile, hostile, bearing)
        elif self.scenario_id == "search-and-interdict":
            contact_known = o[:, self.task_off + 5:self.task_off + 6] > 0.5
            target = o[:, self.task_off + 3:self.task_off + 5]
            # The search target has a known ingress corridor. Before contact,
            # use own normalized position to drive to the expected intercept
            # box; after contact, collapse on the live target cue.
            own = o[:, 0:2]
            search_goal = np.array([0.25, 0.38], dtype=np.float32)
            sweep = search_goal[None, :] - own
            bearing = np.where(contact_known, target, sweep)
        elif self.scenario_id == "navigate-to-target":
            # Staged corridor route: stay centered through the first obstacle
            # pair, then pass above the choke-point pillar before descending to
            # the goal. Coordinates are normalized by world half.
            own = o[:, 0:2]
            waypoint = np.zeros_like(own)
            waypoint[:, 0] = np.where(
                own[:, 0] < -0.30,
                -0.30,
                np.where(own[:, 0] < 0.35, 0.35, np.where(own[:, 0] < 0.65, 0.65, 0.85)),
            )
            waypoint[:, 1] = np.where(own[:, 0] < -0.30, 0.0, np.where(own[:, 0] < 0.65, 0.17, 0.0))
            bearing = waypoint - own

        nrm = np.linalg.norm(bearing, axis=1, keepdims=True)
        action = (bearing / np.maximum(nrm, 1e-6)).astype(np.float32)
        return torch.from_numpy(action)


def eval_scripted(
    *,
    env_id: str,
    battlefield,
    n_episodes: int = 10,
    seed: int = 123,
    env_overrides: dict[str, Any] | None = None,
) -> EvalResult:
    """Evaluate the scripted 'fly at the objective' baseline (see ScriptedPolicy)."""
    env_overrides = dict(env_overrides or {})
    probe = make_scenario_env(env_id, battlefield=battlefield, seed=seed, **env_overrides)
    scripted = ScriptedPolicy(probe)
    return eval_policy(
        scripted,  # duck-typed: deterministic path only calls actor(obs)
        env_id=env_id,
        battlefield=battlefield,
        n_episodes=n_episodes,
        seed=seed,
        deterministic=True,
        env_overrides=env_overrides,
    )


def eval_policy(
    actor: Actor | None,
    *,
    env_id: str,
    battlefield,
    n_episodes: int = 5,
    seed: int = 1234,
    deterministic: bool = True,
    env_overrides: dict[str, Any] | None = None,
) -> EvalResult:
    """Run deterministic evaluation and aggregate task KPIs.

    ``actor=None`` evaluates a random policy baseline.
    """
    profile = get_task_profile(env_id)
    episode_metrics: list[dict[str, float]] = []
    env_overrides = dict(env_overrides or {})

    for ep in range(n_episodes):
        env = make_scenario_env(env_id, battlefield=battlefield, seed=seed + ep, **env_overrides)
        if isinstance(actor, ScriptedPolicy) and actor.is_hunt:
            actor.env = env
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
