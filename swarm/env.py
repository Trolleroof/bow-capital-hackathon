"""Point-mass multi-agent coverage/search environment for the CombatOS swarm.

This is a self-contained vectorized environment (NOT a PettingZoo wrapper) chosen
for a clean, fast MAPPO/CTDE training loop in Phase 1: a shared-parameter actor
consumes the *local* per-agent observation, while a centralized critic consumes
`global_state()` at train time only. Execution stays decentralized — actors never
see global state and send zero messages to each other.

================================================================================
TASK
================================================================================
N point-mass agents move in a bounded 2D square world and must *cover* it. The
world is discretized into a coverage grid; the team is rewarded for newly-covered
cells (search/coverage), lightly penalized for crowding/collisions and for pushing
against the world bounds. z is held at a fixed altitude (the bus message carries z;
coordination is learned in 2D).

================================================================================
OBSERVATION (per agent, LOCAL ONLY — fixed-length float32 vector)
================================================================================
Layout, in order (all positions/velocities normalized to roughly [-1, 1]):

  [ 0: 2 ]   own position (x, y)            normalized to [-1, 1] over the world
             With BattlefieldConfig: GPS denial adds Gaussian noise σ = gps_denial_level×0.2
  [ 2: 4 ]   own velocity (vx, vy)          last applied action, in [-1, 1]
  [ 4: 4+2K] K nearest-neighbor relative positions (dx, dy) each, normalized;
             zero-filled when fewer than K live neighbors exist.
             With BattlefieldConfig: each slot independently zeroed with probability
             jam_duty_cycle (EW jamming — CTDE-safe: actor already handles zero-filled slots)
  [ .. .. ]  local coverage patch: a (PATCH x PATCH) grid centered on the agent,
             flattened row-major. Each cell is 1.0 if that world cell is already
             covered (or out of bounds), else 0.0. Encourages moving toward
             unexplored space.
  [ .. .. ]  nearest M=3 scenario obstacles (boxes / cylinders from
             ``swarm/obstacles.py``): per slot, (dx_to_center/h, dy_to_center/h,
             sx/h, sy/h). Empty slots stay zero. The same registry drives the
             PyBullet renderer, so what the policy is trained against matches
             what you see in 3D.
  [ -1 ]     role/goal flag (float): role index normalized to [0, 1]

Dimensions (defaults N=5, K=3, PATCH=5, M_OBSTACLES=3):
  OWN_DIM        = 4                 (pos 2 + vel 2)
  NEIGHBOR_DIM   = 2 * K             (= 6)
  PATCH_DIM      = PATCH * PATCH     (= 25)
  OBSTACLE_DIM   = M_OBSTACLES * 4   (= 12)
  ROLE_DIM       = 1
  OBS_DIM        = 4 + 6 + 25 + 12 + 1   (= 48)

`obs_dim(K, patch)` recomputes this for non-default configs. The module-level
constant OBS_DIM is for the defaults so Phase 1/2 can import a fixed shape.

================================================================================
ACTION (per agent)
================================================================================
Continuous 2D velocity command in [-1, 1]^2, integrated as point-mass kinematics:
    pos += action * MAX_SPEED * dt   (then clipped to world bounds)
With BattlefieldConfig: wind drift added each step before clip:
    pos += wind_vector * dt          (always applied to live agents)
z is held constant. ACT_DIM = 2.

================================================================================
REWARD (shared / team reward, identical for every live agent)
================================================================================
    + COVERAGE_REWARD  per newly-covered grid cell this step (summed over agents)
    - CROWD_PENALTY    per agent pair closer than CROWD_RADIUS (discourage clumping)
    - EDGE penalty     smooth, distance-from-center term that GROWS as agents push
                       past EDGE_SOFT_FRAC of the world toward the wall. Replaces
                       the old binary "sitting on the edge" penalty so there is a
                       real gradient pulling agents back in before they clip.
    + OBJECTIVE shaping scenario-specific task pull (hold the contested center,
                       shadow the moving target, hold the defend ring, stay in own
                       territory). Gives a non-flat reward landscape once coverage
                       saturates, so agents do the task instead of parking at a wall.
    - COLLISION_PENALTY per agent-step where the body was inside a scenario
                       obstacle's expanded footprint (also gets hard-pushed
                       out the same step, so the policy can never sit inside
                       a wall — it has to plan around it).
Dead agents contribute nothing and receive 0 reward.

================================================================================
ALIVE / KILL
================================================================================
Each agent has an `alive` flag. `kill(agent_id)` freezes the agent (it stops
moving, stops covering cells, and is excluded from neighbor sets) — this drives
the "kill an agent, swarm re-covers the gap with zero comms" money demo.
With BattlefieldConfig: `attrition_inject_rate` triggers random kills each step.

================================================================================
BATTLEFIELD PARAMETERS (see swarm/env_config.py and docs/battlefield-parameters.md)
================================================================================
Pass a BattlefieldConfig to SwarmEnv to activate P0 parameters:
  - wind_speed / wind_dir_rad : drift added to position integration
  - gps_denial_level          : Gaussian noise on obs[0:2] (own position)
  - jam_duty_cycle            : per-slot neighbor dropout in obs[4:4+2K]
  - attrition_inject_rate     : per-step probability of a random agent kill
  - battery_envelope_sec /
    time_limit_sec            : sets max_steps (via BattlefieldConfig.max_steps)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict

import numpy as np

try:
    from .env_config import BattlefieldConfig
    from .obstacles import Obstacle, obstacles_for
except ImportError:  # when run as __main__ directly (python env.py)
    from env_config import BattlefieldConfig  # type: ignore[no-redef]
    from obstacles import Obstacle, obstacles_for  # type: ignore[no-redef]

# ------------------------------------------------------------------ defaults ---
N_AGENTS = 5
K_NEIGHBORS = 3
PATCH = 5            # local coverage patch is PATCH x PATCH cells, centered on agent
GRID = 20            # world coverage grid is GRID x GRID cells
WORLD_HALF = 10.0    # world spans [-WORLD_HALF, WORLD_HALF] in x and y
ALTITUDE = 2.0       # fixed z (point-mass; z is not learned)

DT = 0.1
MAX_SPEED = 6.0      # world-units / second at full throttle

COVERAGE_REWARD = 1.0
CROWD_PENALTY = 0.05
CROWD_RADIUS = 1.5

# ── boundary handling (issue: agents drift out and park at the wall) ─────────
# Old behaviour penalized only agents *already sitting on* the edge with a tiny
# weight (0.02), ~50x smaller than a single covered cell, and gave no gradient
# as agents approached the wall. We instead apply a smooth penalty that grows
# quadratically once an agent passes EDGE_SOFT_FRAC of the world radius, so the
# policy feels increasing drag the closer it gets to the boundary.
EDGE_SOFT_FRAC = 0.8          # soft boundary at 80% of world_half
EDGE_GRADIENT_PENALTY = 0.4   # weight on (distance-beyond-soft)^2, summed over live

# ── scenario objective shaping (issue: no per-scenario task / objects) ───────
# Every scenario shares the coverage env, so without this all policies collapse
# to the same wall-hugging coverage behaviour. This adds a small per-step pull
# toward each scenario's objective so behaviour differs and survives coverage
# saturation. Kept well below the coverage scale so exploration still dominates
# early in the episode.
OBJECTIVE_REWARD = 0.06       # weight on normalized distance-to-objective

# ── scenario obstacles (real 3D props that the policy now sees + avoids) ─────
# Each obstacle contributes a 4-float slot to the per-agent observation
# (relative dx, dy to obstacle center and its half-extents, all normalized).
# Push-out is hard: agents physically cannot end a step inside an obstacle's
# expanded footprint. A small per-collision penalty keeps the gradient honest.
M_OBSTACLES = 3                 # how many nearest obstacles enter local obs
OBSTACLE_FEATS = 4              # dx/h, dy/h, sx/h, sy/h
OBSTACLE_DIM = M_OBSTACLES * OBSTACLE_FEATS
AGENT_RADIUS = 0.4              # body half-extent used in collision push-out
COLLISION_PENALTY = 0.4         # per-agent-step penalty for trying to enter

# ── drone-vs-drone combat shaping ────────────────────────────────────────────
HOSTILE_ENGAGE_RADIUS = 2.0
HOSTILE_KILL_REWARD = 2.5
HOSTILE_APPROACH_REWARD = 0.05
DRONE_VS_DRONE_COVERAGE_SCALE = 0.12
HOVER_CENTER_FRAC = 0.32
HOVER_SPEED_MAX = 0.28
HOVER_REWARD = 0.12
OBJECTIVE_REWARD_POST_KILL = 0.18

ROLES = ("scout", "scout", "scout", "scout", "scout")  # all scouts for Phase 0


def obs_dim(k: int = K_NEIGHBORS, patch: int = PATCH, m_obstacles: int = M_OBSTACLES) -> int:
    """Compute the local observation vector length for a given K / patch size."""
    return 4 + 2 * k + patch * patch + m_obstacles * OBSTACLE_FEATS + 1


# Importable fixed dims for the default config (Phase 1/2 import these).
OWN_DIM = 4
NEIGHBOR_DIM = 2 * K_NEIGHBORS
PATCH_DIM = PATCH * PATCH
ROLE_DIM = 1
OBS_DIM = obs_dim()          # 48 for defaults (was 36 pre-obstacles)
ACT_DIM = 2


class SwarmEnv:
    """Vectorized point-mass coverage env. See module docstring for full spec."""

    def __init__(
        self,
        n_agents: int = N_AGENTS,
        k_neighbors: int = K_NEIGHBORS,
        patch: int = PATCH,
        grid: int = GRID,
        world_half: float = WORLD_HALF,
        max_steps: int = 400,
        seed: int | None = None,
        battlefield: BattlefieldConfig | None = None,
        scenario_id: str | None = None,
        obstacles: list[Obstacle] | None = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Battlefield config — P0 parameters (wind, EW, attrition, limits).
        # When provided, BattlefieldConfig.max_steps and n_agents take precedence
        # over the positional arguments so callers using make_scenario_env get
        # consistent behaviour.
        # ------------------------------------------------------------------
        self.battlefield: BattlefieldConfig | None = battlefield
        if battlefield is not None:
            n_agents = battlefield.n_agents
            max_steps = battlefield.max_steps

        # Scenario id drives spawn regions + objective shaping (purely affects
        # spawn positions and reward; obs/act dims are unchanged for portability).
        self.scenario_id = scenario_id
        self.n = n_agents
        self.k = k_neighbors
        self.patch = patch
        self.grid = grid
        self.world_half = world_half
        self.max_steps = max_steps
        self.obs_dim = obs_dim(k_neighbors, patch)
        self.act_dim = ACT_DIM
        self.cell = (2.0 * world_half) / grid  # world units per grid cell
        self.rng = np.random.default_rng(seed)

        # Pre-compute wind drift vector (world-units per second) from config.
        # Applied as:  pos += _wind * DT  each step for live agents.
        self._wind = np.zeros(2, dtype=np.float32)
        if battlefield is not None:
            ws = battlefield.weather.wind_speed
            wd = battlefield.weather.wind_dir_rad
            self._wind[0] = ws * math.cos(wd)
            self._wind[1] = ws * math.sin(wd)

        # state (filled by reset)
        self.pos = np.zeros((self.n, 2), dtype=np.float32)
        self.vel = np.zeros((self.n, 2), dtype=np.float32)
        self.alive = np.ones(self.n, dtype=bool)
        self.covered = np.zeros((grid, grid), dtype=bool)
        self.roles = np.zeros(self.n, dtype=np.int64)
        self.t = 0
        self.steps = 0
        self.hostile_pos = np.zeros((0, 2), dtype=np.float32)
        self.hostile_alive = np.zeros(0, dtype=bool)

        # Scenario obstacles — real 3D props the policy must see + avoid.
        # Default to the registry for this scenario_id; can be overridden for tests.
        if obstacles is None and scenario_id is not None:
            obstacles = obstacles_for(scenario_id)
        self.obstacles: list[Obstacle] = list(obstacles or [])
        if self.obstacles:
            self._obs_centers = np.array(
                [[o.cx, o.cy] for o in self.obstacles], dtype=np.float32
            )
            self._obs_half = np.array(
                [[o.sx, o.sy] for o in self.obstacles], dtype=np.float32
            )
            self._obs_is_circle = np.array(
                [o.kind == "cylinder" for o in self.obstacles], dtype=bool
            )
        else:
            self._obs_centers = np.zeros((0, 2), dtype=np.float32)
            self._obs_half = np.zeros((0, 2), dtype=np.float32)
            self._obs_is_circle = np.zeros(0, dtype=bool)

    def _hostile_count(self) -> int:
        if self.scenario_id != "drone-vs-drone":
            return 0
        if self.battlefield is not None:
            return max(1, int(self.battlefield.threat.hostile_uas_count))
        return 3

    def _reset_hostiles(self) -> None:
        n_hostile = self._hostile_count()
        if n_hostile == 0:
            self.hostile_pos = np.zeros((0, 2), dtype=np.float32)
            self.hostile_alive = np.zeros(0, dtype=bool)
            return
        h = self.world_half
        x = self.rng.uniform(0.6 * h, 0.92 * h, n_hostile)
        y = self.rng.uniform(-0.7 * h, 0.7 * h, n_hostile)
        self.hostile_pos = np.stack([x, y], axis=1).astype(np.float32)
        self.hostile_alive = np.ones(n_hostile, dtype=bool)

    def _update_hostiles(self) -> None:
        """Patrol the contested center lane (matches PyBullet scripted reds)."""
        if self.hostile_alive.size == 0:
            return
        h = self.world_half
        cx, cy = 0.48 * h, 0.0
        for i in range(self.hostile_alive.size):
            if not self.hostile_alive[i]:
                continue
            r = 0.24 * h * (2.4 + i * 0.8)
            ang = self.t * 0.5 + i * 2.1
            self.hostile_pos[i, 0] = cx + r * math.cos(ang) * 0.85
            self.hostile_pos[i, 1] = cy + r * math.sin(ang) * 0.85

    def _drone_vs_drone_combat_reward(self, live_pos: np.ndarray, live: np.ndarray) -> float:
        if live_pos.shape[0] == 0 or self.hostile_alive.size == 0:
            return 0.0

        reward = 0.0
        h = self.world_half
        live_hostile = self.hostile_pos[self.hostile_alive]
        if live_hostile.shape[0] > 0:
            dists = np.linalg.norm(
                live_pos[:, None, :] - live_hostile[None, :, :],
                axis=-1,
            )
            nearest = dists.min(axis=1) / (2.0 * h)
            reward += HOSTILE_APPROACH_REWARD * float((1.0 - nearest).sum())

            for idx in np.where(self.hostile_alive)[0]:
                if np.any(np.linalg.norm(live_pos - self.hostile_pos[idx], axis=1) < HOSTILE_ENGAGE_RADIUS):
                    self.hostile_alive[idx] = False
                    reward += HOSTILE_KILL_REWARD

        if not self.hostile_alive.any():
            dist_center = np.linalg.norm(live_pos, axis=1) / h
            in_center = dist_center < HOVER_CENTER_FRAC
            speeds = np.linalg.norm(self.vel[live], axis=1)
            slow = speeds < HOVER_SPEED_MAX
            reward += HOVER_REWARD * float((in_center & slow).sum()) / max(1, int(live.sum()))
            reward += -OBJECTIVE_REWARD_POST_KILL * float(dist_center.mean())
        return reward
    def _resolve_obstacle_collisions(self, live_mask: np.ndarray) -> int:
        """Hard push every live agent out of any obstacle's expanded footprint.

        Each obstacle footprint is grown by AGENT_RADIUS, then any agent inside
        is shoved to the nearest edge (boxes) or onto the inflated circle
        (cylinders). Returns the number of live agents that needed correction
        this step — used as a collision count for the reward.
        """
        if not self.obstacles or not live_mask.any():
            return 0
        live_idx = np.where(live_mask)[0]
        collided = 0
        for i in live_idx:
            px, py = float(self.pos[i, 0]), float(self.pos[i, 1])
            hit_any = False
            # Two passes so an agent pushed out of one obstacle can't still sit
            # inside an adjacent one (rare with our layouts but cheap insurance).
            for _ in range(2):
                still_hit = False
                for k, obs in enumerate(self.obstacles):
                    cx, cy = float(obs.cx), float(obs.cy)
                    if self._obs_is_circle[k]:
                        r = float(obs.sx) + AGENT_RADIUS
                        dx, dy = px - cx, py - cy
                        d2 = dx * dx + dy * dy
                        if d2 < r * r:
                            d = math.sqrt(d2) if d2 > 1e-12 else 1e-6
                            # pick a deterministic direction when at exact center
                            if d < 1e-5:
                                dx, dy, d = 1.0, 0.0, 1.0
                            px = cx + dx / d * r
                            py = cy + dy / d * r
                            still_hit = True
                            hit_any = True
                    else:
                        hx = float(obs.sx) + AGENT_RADIUS
                        hy = float(obs.sy) + AGENT_RADIUS
                        if abs(px - cx) < hx and abs(py - cy) < hy:
                            # push along the axis of least penetration
                            pen_x = hx - abs(px - cx)
                            pen_y = hy - abs(py - cy)
                            if pen_x < pen_y:
                                px = cx + math.copysign(hx, px - cx if px != cx else 1.0)
                            else:
                                py = cy + math.copysign(hy, py - cy if py != cy else 1.0)
                            still_hit = True
                            hit_any = True
                if not still_hit:
                    break
            if hit_any:
                self.pos[i, 0] = px
                self.pos[i, 1] = py
                collided += 1
        return collided

    def _world_to_cell(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map world (x, y) -> integer grid indices (cx, cy), clipped to grid."""
        idx = ((p + self.world_half) / self.cell).astype(np.int64)
        idx = np.clip(idx, 0, self.grid - 1)
        return idx[..., 0], idx[..., 1]

    def _mark_covered(self) -> int:
        """Mark cells under live agents as covered; return # newly covered."""
        before = int(self.covered.sum())
        if self.alive.any():
            cx, cy = self._world_to_cell(self.pos[self.alive])
            self.covered[cx, cy] = True
        return int(self.covered.sum()) - before

    # ------------------------------------------------------ scenario shaping ---
    def _spawn_positions(self) -> np.ndarray:
        """Scenario-specific spawn areas (world units), not a center cluster.

        Each scenario stages the swarm from a plausible starting posture for its
        task so they begin spread out and oriented toward the objective.
        """
        h = self.world_half
        n = self.n
        rng = self.rng
        sid = self.scenario_id

        if sid == "drone-vs-drone":
            # blue team pushes in from the left flank, spread along that edge
            x = rng.uniform(-0.92 * h, -0.6 * h, n)
            y = rng.uniform(-0.7 * h, 0.7 * h, n)
        elif sid == "moving-target-track":
            # trackers fan out across the field to pick up the weaving mover
            x = rng.uniform(-0.8 * h, 0.8 * h, n)
            y = rng.uniform(-0.8 * h, 0.8 * h, n)
        elif sid == "search-and-interdict":
            # sweep team enters along the bottom edge and works upward
            x = rng.uniform(-0.85 * h, 0.85 * h, n)
            y = rng.uniform(-0.92 * h, -0.6 * h, n)
        elif sid == "defend-asset":
            # defenders start on a ring around the central asset
            ang = rng.uniform(0.0, 2.0 * math.pi, n)
            r = rng.uniform(0.42 * h, 0.55 * h, n)
            x = r * np.cos(ang)
            y = r * np.sin(ang)
        elif sid == "swarm-vs-swarm-race":
            # blue holds and works the left territory
            x = rng.uniform(-0.9 * h, -0.15 * h, n)
            y = rng.uniform(-0.8 * h, 0.8 * h, n)
        else:
            # default: spread across the arena (no tiny center cluster)
            x = rng.uniform(-0.75 * h, 0.75 * h, n)
            y = rng.uniform(-0.75 * h, 0.75 * h, n)

        return np.stack([x, y], axis=1).astype(np.float32)

    def _objective_point(self) -> np.ndarray | None:
        """Current objective location (world units) for point/track scenarios.

        Moving scenarios derive the point analytically from ``self.t`` using the
        same formulas the PyBullet renderer scripts, so the trained behaviour and
        the rendered target/mover stay visually consistent.
        """
        h = self.world_half
        sid = self.scenario_id
        if sid == "drone-vs-drone":
            return np.array([0.0, 0.0], dtype=np.float32)  # contested center lane
        if sid == "moving-target-track":
            t = self.t
            return np.array(
                [0.6 * h * math.sin(t * 0.22), 0.45 * h * math.sin(t * 0.41 + 0.6)],
                dtype=np.float32,
            )
        if sid == "search-and-interdict":
            t = self.t
            return np.array(
                [0.5 * h * math.sin(t * 0.18) + 0.15 * h, 0.5 * h * math.cos(t * 0.27)],
                dtype=np.float32,
            )
        if sid == "swarm-vs-swarm-race":
            return np.array([-0.5 * h, 0.0], dtype=np.float32)  # own territory
        return None

    def _objective_reward(self, live_pos: np.ndarray) -> float:
        """Per-step scenario task pull (already restricted to live agents)."""
        if live_pos.shape[0] == 0:
            return 0.0
        h = self.world_half
        if self.scenario_id == "defend-asset":
            # reward holding a standoff ring around the asset (peak on the ring)
            ring = 0.6 * h
            dist = np.linalg.norm(live_pos, axis=1)
            band = np.abs(dist - ring) / h
            return -OBJECTIVE_REWARD * float(band.mean())
        pt = self._objective_point()
        if pt is None:
            return 0.0
        dist = np.linalg.norm(live_pos - pt, axis=1) / (2.0 * h)  # ~[0, 0.7]
        return -OBJECTIVE_REWARD * float(dist.mean())

    # ------------------------------------------------------------------ reset ---
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset world and return per-agent local obs, shape (n, obs_dim)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Spawn from a scenario-specific staging area (a flank, an edge, a ring,
        # or spread across the field) instead of one tiny cluster at the center.
        self.pos = self._spawn_positions()
        self.vel = np.zeros((self.n, 2), dtype=np.float32)
        self.alive = np.ones(self.n, dtype=bool)
        self.covered = np.zeros((self.grid, self.grid), dtype=bool)
        self.roles = np.zeros(self.n, dtype=np.int64)
        self.t = 0.0
        self.steps = 0
        self._reset_hostiles()
        self._mark_covered()
        return self._obs()

    # ------------------------------------------------------------------- step ---
    def step(self, actions: np.ndarray):
        """Integrate point-mass kinematics for one tick.

        actions: (n, 2) float in [-1, 1]. Returns (obs, rewards, dones, info)
        where obs is (n, obs_dim), rewards is (n,) shared team reward (0 for dead
        agents), dones is (n,) bool, info is a dict.
        """
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n, 2)
        actions = np.clip(actions, -1.0, 1.0)

        # ── Battlefield: random attrition (P0) ───────────────────────────
        if (
            self.battlefield is not None
            and self.battlefield.logistics.attrition_inject_rate > 0.0
        ):
            rate = self.battlefield.logistics.attrition_inject_rate
            live_ids = np.where(self.alive)[0]
            for aid in live_ids:
                if self.rng.random() < rate:
                    self.kill(aid)

        live = self.alive
        self.vel = actions  # record applied command (used in obs)
        # only live agents move
        self.pos[live] += actions[live] * MAX_SPEED * DT

        # ── Battlefield: wind drift (P0) ─────────────────────────────────
        # Applied after agent command so policy must compensate up-wind.
        if self.battlefield is not None and np.any(self._wind != 0):
            self.pos[live] += self._wind * DT

        bound = self.world_half
        self.pos = np.clip(self.pos, -bound, bound)

        # Hard collision push-out against scenario obstacles, then re-clip in
        # case the push moved an agent past the world wall.
        n_collisions = self._resolve_obstacle_collisions(live)
        if n_collisions:
            self.pos = np.clip(self.pos, -bound, bound)

        new_cells = self._mark_covered()
        self._update_hostiles()

        coverage_scale = (
            DRONE_VS_DRONE_COVERAGE_SCALE
            if self.scenario_id == "drone-vs-drone"
            else 1.0
        )
        reward = COVERAGE_REWARD * coverage_scale * new_cells

        live_pos = self.pos[live]

        # crowding: penalize close live pairs
        if live.sum() > 1:
            d = np.linalg.norm(live_pos[:, None, :] - live_pos[None, :, :], axis=-1)
            iu = np.triu_indices(live_pos.shape[0], k=1)
            reward -= CROWD_PENALTY * int((d[iu] < CROWD_RADIUS).sum())

        if live_pos.shape[0] > 0:
            # smooth boundary penalty: grows as agents push past the soft radius,
            # so there is a real gradient pulling them back before they clip.
            dist_c = np.linalg.norm(live_pos, axis=1) / bound        # ~[0, 1.41]
            excess = np.clip(dist_c - EDGE_SOFT_FRAC, 0.0, None)
            edge_weight = 1.8 if self.scenario_id == "drone-vs-drone" else 1.0
            reward -= EDGE_GRADIENT_PENALTY * edge_weight * float((excess ** 2).sum())

            if self.scenario_id == "drone-vs-drone":
                reward += self._drone_vs_drone_combat_reward(live_pos, live)
                if self.hostile_alive.any():
                    reward += 0.5 * self._objective_reward(live_pos)
            else:
                reward += self._objective_reward(live_pos)

        if n_collisions:
            reward -= COLLISION_PENALTY * float(n_collisions)

        self.steps += 1
        self.t += DT
        done = self.steps >= self.max_steps

        rewards = np.where(self.alive, np.float32(reward), np.float32(0.0)).astype(
            np.float32
        )
        dones = np.full(self.n, done, dtype=bool)
        info: dict = {
            "new_cells": new_cells,
            "coverage": self.coverage_fraction(),
            "n_alive": int(self.alive.sum()),
            "n_collisions": n_collisions,
        }
        if self.hostile_alive.size:
            info["n_hostile_alive"] = int(self.hostile_alive.sum())
        # Log params hash so training checkpoints are traceable (issue #15).
        if self.battlefield is not None:
            info["params_hash"] = self.battlefield_hash()
        return self._obs(), rewards, dones, info

    # ----------------------------------------------------------- observations ---
    def _obs(self) -> np.ndarray:
        """Build the (n, obs_dim) local observation matrix. See module docstring."""
        out = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        half = self.world_half
        norm_pos = self.pos / half  # ~[-1, 1]

        # ── Battlefield: EW parameter shortcuts ─────────────────────────
        gps_noise_sigma = (
            self.battlefield.ew.gps_denial_level * 0.2
            if self.battlefield is not None else 0.0
        )
        jam_duty_cycle = (
            self.battlefield.ew.jam_duty_cycle
            if self.battlefield is not None else 0.0
        )

        for i in range(self.n):
            o = 0

            # [0:2] own position (normalized) + GPS denial noise (P0)
            pos_obs = norm_pos[i].copy()
            if gps_noise_sigma > 0.0:
                pos_obs += self.rng.standard_normal(2).astype(np.float32) * gps_noise_sigma
            out[i, o:o + 2] = pos_obs; o += 2

            out[i, o:o + 2] = self.vel[i]; o += 2          # own velocity (cmd)

            # K nearest LIVE neighbors (relative position), zero-filled.
            # Jamming (P0): each neighbor slot zeroed independently with prob jam_duty_cycle.
            others = [j for j in range(self.n) if j != i and self.alive[j]]
            if others:
                rel = self.pos[others] - self.pos[i]
                dist = np.linalg.norm(rel, axis=1)
                order = np.argsort(dist)[: self.k]
                for n_idx in order:
                    if jam_duty_cycle > 0.0 and self.rng.random() < jam_duty_cycle:
                        o += 2  # slot zeroed (already 0 from np.zeros init)
                    else:
                        out[i, o:o + 2] = rel[n_idx] / half
                        o += 2
                o = 4 + 2 * self.k  # advance past any unfilled neighbor slots
            else:
                o = 4 + 2 * self.k

            # local coverage patch centered on this agent (PATCH x PATCH)
            cx, cy = self._world_to_cell(self.pos[i][None, :])
            cx, cy = int(cx[0]), int(cy[0])
            r = self.patch // 2
            patch = np.ones((self.patch, self.patch), dtype=np.float32)  # OOB = 1
            for a in range(self.patch):
                gx = cx - r + a
                if gx < 0 or gx >= self.grid:
                    continue
                for b in range(self.patch):
                    gy = cy - r + b
                    if 0 <= gy < self.grid:
                        patch[a, b] = 1.0 if self.covered[gx, gy] else 0.0
            out[i, o:o + self.patch * self.patch] = patch.reshape(-1)
            o += self.patch * self.patch

            # nearest M_OBSTACLES scenario obstacles, each as
            # (dx_to_center/h, dy_to_center/h, sx/h, sy/h). Slots beyond the
            # available obstacle count stay zero, which the policy reads as
            # "no further obstacle nearby".
            if self._obs_centers.shape[0] > 0:
                rel = self._obs_centers - self.pos[i]
                d = np.linalg.norm(rel, axis=1)
                order = np.argsort(d)[:M_OBSTACLES]
                for slot_i, k in enumerate(order):
                    base = o + slot_i * OBSTACLE_FEATS
                    out[i, base + 0] = rel[k, 0] / half
                    out[i, base + 1] = rel[k, 1] / half
                    out[i, base + 2] = self._obs_half[k, 0] / half
                    out[i, base + 3] = self._obs_half[k, 1] / half
            o += OBSTACLE_DIM

            # role flag normalized to [0, 1]
            out[i, o] = self.roles[i] / max(1, len(ROLES) - 1)
        return out

    def global_state(self) -> np.ndarray:
        """Centralized-critic state (train-only, Phase 1).

        Concatenates all agents' positions, velocities, alive flags, plus the
        flattened global coverage grid. NOT used by the decentralized actor.

        With BattlefieldConfig: appends 4 normalized P0 scalars at the end
        [wind_speed/15, jam_duty_cycle, gps_denial_level, attrition_inject_rate/0.5]
        so the centralized critic can condition on environment stress during training.
        These are NEVER in the actor's local obs (CTDE-safe).
        """
        parts = [
            (self.pos / self.world_half).reshape(-1),
            self.vel.reshape(-1),
            self.alive.astype(np.float32),
            self.covered.astype(np.float32).reshape(-1),
        ]
        if self.battlefield is not None:
            bf = self.battlefield
            parts.append(np.array([
                bf.weather.wind_speed / 15.0,
                bf.ew.jam_duty_cycle,
                bf.ew.gps_denial_level,
                bf.logistics.attrition_inject_rate / 0.5,
            ], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    @property
    def state_dim(self) -> int:
        base = self.n * 2 + self.n * 2 + self.n + self.grid * self.grid
        if self.battlefield is not None:
            base += 4  # 4 normalized P0 scalars appended to global state
        return base

    def battlefield_hash(self) -> str:
        """SHA-256 of the battlefield config JSON, first 12 hex chars.

        Logged alongside checkpoints so a training run is always traceable to
        its parameter set (issue #15 acceptance criterion).
        """
        if self.battlefield is None:
            return "garrison"
        raw = json.dumps(asdict(self.battlefield), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    # ----------------------------------------------------------------- extras ---
    def coverage_fraction(self) -> float:
        return float(self.covered.sum()) / float(self.grid * self.grid)

    def kill(self, agent_id: int) -> None:
        """Kill an agent: it freezes, stops covering, and leaves neighbor sets."""
        self.alive[agent_id] = False
        self.vel[agent_id] = 0.0

    def revive(self, agent_id: int) -> None:
        self.alive[agent_id] = True


if __name__ == "__main__":
    # tiny smoke test
    env = SwarmEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (env.n, OBS_DIM), (obs.shape, OBS_DIM)
    for t in range(500):
        a = env.rng.uniform(-1, 1, size=(env.n, 2)).astype(np.float32)
        obs, r, d, info = env.step(a)
        if t == 250:
            env.kill(0)
    print(
        f"OK obs={obs.shape} state_dim={env.state_dim} "
        f"coverage={info['coverage']:.2f} alive={info['n_alive']}"
    )
