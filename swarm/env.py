"""Point-mass multi-agent coverage/search environment for the Outcast Virus swarm.

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
  [ .. .. ]  scenario task features: targets, hostiles, rivals, asset cues.
  [ -1 ]     role/goal flag (float): role index normalized to [0, 1]

Dimensions (defaults N=5, K=3, PATCH=5, M_OBSTACLES=3):
  OWN_DIM        = 4                 (pos 2 + vel 2)
  NEIGHBOR_DIM   = 2 * K             (= 6)
  PATCH_DIM      = PATCH * PATCH     (= 25)
  OBSTACLE_DIM   = M_OBSTACLES * 4   (= 12)
  TASK_DIM       = 16
  ROLE_DIM       = 1
  OBS_DIM        = 4 + 6 + 25 + 12 + 16 + 1   (= 64)

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
    from .task_profiles import TASK_DIM, TaskProfile, get_task_profile
except ImportError:  # when run as __main__ directly (python env.py)
    from env_config import BattlefieldConfig  # type: ignore[no-redef]
    from obstacles import Obstacle, obstacles_for  # type: ignore[no-redef]
    from task_profiles import TASK_DIM, TaskProfile, get_task_profile  # type: ignore[no-redef]

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
MIN_AGENT_SEPARATION = 1.25
DECONFLICT_RADIUS = 2.6
SEPARATION_STEER = 1.1

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
    return 4 + 2 * k + patch * patch + m_obstacles * OBSTACLE_FEATS + TASK_DIM + 1


# Importable fixed dims for the default config (Phase 1/2 import these).
OWN_DIM = 4
NEIGHBOR_DIM = 2 * K_NEIGHBORS
PATCH_DIM = PATCH * PATCH
ROLE_DIM = 1
BASE_WITHOUT_ROLE_DIM = OWN_DIM + NEIGHBOR_DIM + PATCH_DIM + OBSTACLE_DIM
OBS_DIM = obs_dim()          # 64 for defaults (48 base + 16 task feats)
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
        self.task_profile: TaskProfile = get_task_profile(scenario_id)
        self.task_phase = self.task_profile.phase_names[0] if self.task_profile.phase_names else "coverage"
        self.task_events: dict[str, int | float | bool] = {}
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.target_vel = np.zeros(2, dtype=np.float32)
        self.hostile_pos = np.zeros((0, 2), dtype=np.float32)
        self.hostile_vel = np.zeros((0, 2), dtype=np.float32)
        self.hostile_alive = np.zeros(0, dtype=bool)
        self.rival_pos = np.zeros((0, 2), dtype=np.float32)
        self.rival_vel = np.zeros((0, 2), dtype=np.float32)
        self.contested_cells = np.zeros((0, 2), dtype=np.int64)
        self.contested_owner = np.zeros(0, dtype=np.int8)
        self.asset_pos = np.zeros(2, dtype=np.float32)
        self.breaches = 0
        self.intercepts = 0
        self.custody_steps = 0
        self.lost_track_steps = 0
        self.contact_step: int | None = None
        self.intercept_step: int | None = None
        self.contested_score = 0
        self.rival_score = 0
        self._nav_prev_x: float = -self.world_half    # navigate-to-target x-progress tracker
        self._nav_prev_dist: float = 2.0 * self.world_half  # navigate-to-target dist tracker
        self._d2d_prev_dist: float = 2.0 * self.world_half  # drone-vs-drone progress tracker

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
        if self.scenario_id not in {"drone-vs-drone", "defend-asset"}:
            return 0
        if self.battlefield is not None:
            return max(1, int(self.battlefield.threat.hostile_uas_count))
        return 3 if self.scenario_id == "drone-vs-drone" else 5

    def _reset_task_entities(self) -> None:
        h = self.world_half
        self.task_phase = self.task_profile.phase_names[0] if self.task_profile.phase_names else "coverage"
        self.task_events = {}
        self.target_pos = np.array([0.45 * h, 0.35 * h], dtype=np.float32)
        self.target_vel = np.zeros(2, dtype=np.float32)
        self.hostile_pos = np.zeros((0, 2), dtype=np.float32)
        self.hostile_vel = np.zeros((0, 2), dtype=np.float32)
        self.hostile_alive = np.zeros(0, dtype=bool)
        self.rival_pos = np.zeros((0, 2), dtype=np.float32)
        self.rival_vel = np.zeros((0, 2), dtype=np.float32)
        self.contested_cells = np.zeros((0, 2), dtype=np.int64)
        self.contested_owner = np.zeros(0, dtype=np.int8)
        self.asset_pos = np.zeros(2, dtype=np.float32)
        self.breaches = 0
        self.intercepts = 0
        self.custody_steps = 0
        self.lost_track_steps = 0
        self.contact_step = None
        self.intercept_step = None
        self.contested_score = 0
        self.rival_score = 0

        sid = self.scenario_id
        if sid == "drone-vs-drone":
            n_hostile = self._hostile_count()
            x = self.rng.uniform(0.35 * h, 0.85 * h, n_hostile)
            y = self.rng.uniform(-0.65 * h, 0.65 * h, n_hostile)
            self.hostile_pos = np.stack([x, y], axis=1).astype(np.float32)
            self.hostile_vel = np.zeros((n_hostile, 2), dtype=np.float32)
            self.hostile_alive = np.ones(n_hostile, dtype=bool)
            self._d2d_prev_dist = self._drone_vs_drone_mean_nearest_dist()
        elif sid == "moving-target-track":
            self.target_pos = np.array([-0.45 * h, -0.15 * h], dtype=np.float32)
            speed = 0.6
            if self.battlefield is not None:
                speed = max(0.15, float(self.battlefield.threat.moving_target_speed))
            self.target_vel = np.array([speed, speed * 0.45], dtype=np.float32)
        elif sid == "search-and-interdict":
            self.target_pos = np.array([0.35 * h, 0.45 * h], dtype=np.float32)
            self.target_vel = np.array([-0.18, -0.08], dtype=np.float32)
        elif sid == "defend-asset":
            n_hostile = self._hostile_count()
            angles = np.linspace(0.0, 2.0 * math.pi, n_hostile, endpoint=False)
            radius = 0.95 * h
            self.hostile_pos = np.stack(
                [radius * np.cos(angles), radius * np.sin(angles)], axis=1
            ).astype(np.float32)
            inward = -self.hostile_pos / np.maximum(
                np.linalg.norm(self.hostile_pos, axis=1, keepdims=True), 1e-6
            )
            self.hostile_vel = (inward * 0.55).astype(np.float32)
            self.hostile_alive = np.ones(n_hostile, dtype=bool)
        elif sid == "navigate-to-target":
            # Static goal at the right end of the corridor.
            self.target_pos = np.array([0.85 * h, 0.0], dtype=np.float32)
            self.target_vel = np.zeros(2, dtype=np.float32)
            # Track x and dist for per-step progress shaping.
            self._nav_prev_x = float(self.pos[0, 0]) if self.pos.shape[0] > 0 else -h
            self._nav_prev_dist = float(np.linalg.norm(self.pos[0] - self.target_pos)) if self.pos.shape[0] > 0 else 2.0 * h

    def _update_task_entities(self) -> None:
        h = self.world_half
        sid = self.scenario_id
        if sid == "drone-vs-drone" and self.hostile_alive.size:
            center = np.array([0.45 * h, 0.0], dtype=np.float32)
            for i in range(self.hostile_alive.size):
                if not self.hostile_alive[i]:
                    continue
                radius = 0.16 * h + i * 0.08 * h
                ang = self.t * 0.45 + i * 2.1
                new_pos = center + np.array(
                    [math.cos(ang) * radius, math.sin(ang) * radius], dtype=np.float32
                )
                self.hostile_vel[i] = (new_pos - self.hostile_pos[i]) / DT
                self.hostile_pos[i] = new_pos
        elif sid == "moving-target-track":
            speed = 0.6
            if self.battlefield is not None:
                speed = max(0.15, float(self.battlefield.threat.moving_target_speed))
            new_pos = np.array(
                [
                    0.62 * h * math.sin(self.t * 0.20 * speed),
                    0.45 * h * math.sin(self.t * 0.37 * speed + 0.7),
                ],
                dtype=np.float32,
            )
            self.target_vel = (new_pos - self.target_pos) / DT
            self.target_pos = new_pos
        elif sid == "search-and-interdict":
            new_pos = np.array(
                [
                    0.42 * h * math.sin(self.t * 0.16) + 0.18 * h,
                    0.42 * h * math.cos(self.t * 0.23),
                ],
                dtype=np.float32,
            )
            self.target_vel = (new_pos - self.target_pos) / DT
            self.target_pos = new_pos
        elif sid == "defend-asset" and self.hostile_alive.size:
            for i in range(self.hostile_alive.size):
                if not self.hostile_alive[i]:
                    continue
                self.hostile_pos[i] += self.hostile_vel[i] * DT

    def _nearest_live_hostile(self, point: np.ndarray) -> tuple[int | None, float]:
        if self.hostile_alive.size == 0 or not self.hostile_alive.any():
            return None, float("inf")
        ids = np.where(self.hostile_alive)[0]
        d = np.linalg.norm(self.hostile_pos[ids] - point, axis=1)
        k = int(np.argmin(d))
        return int(ids[k]), float(d[k])

    def _drone_vs_drone_mean_nearest_dist(self) -> float:
        if not self.alive.any() or self.hostile_alive.size == 0 or not self.hostile_alive.any():
            return 2.0 * self.world_half
        live_pos = self.pos[self.alive]
        live_hostile = self.hostile_pos[self.hostile_alive]
        dist_to_hostiles = np.linalg.norm(
            live_pos[:, None, :] - live_hostile[None, :, :],
            axis=-1,
        )
        return float(dist_to_hostiles.min(axis=1).mean())

    def _compute_task_transitions(self, live_pos: np.ndarray) -> dict[str, float]:
        events: dict[str, float] = {}
        sid = self.scenario_id
        if live_pos.shape[0] == 0:
            return events

        if sid in {"drone-vs-drone", "defend-asset"} and self.hostile_alive.size:
            for idx in np.where(self.hostile_alive)[0]:
                d = np.linalg.norm(live_pos - self.hostile_pos[idx], axis=1)
                if np.any(d < HOSTILE_ENGAGE_RADIUS):
                    self.hostile_alive[idx] = False
                    self.intercepts += 1
                    events["kills"] = events.get("kills", 0.0) + 1.0

        if sid == "drone-vs-drone":
            self.task_phase = "orbit" if self.hostile_alive.size and not self.hostile_alive.any() else "engage"
        elif sid == "moving-target-track":
            d = np.linalg.norm(live_pos - self.target_pos, axis=1)
            in_custody = d < 2.8
            if np.count_nonzero(in_custody) >= max(1, min(2, live_pos.shape[0])):
                self.custody_steps += 1
                events["custody"] = 1.0
            else:
                self.lost_track_steps += 1
                events["lost_track"] = 1.0
        elif sid == "search-and-interdict":
            d = np.linalg.norm(live_pos - self.target_pos, axis=1)
            if self.contact_step is None and float(d.min()) < 4.2:
                self.contact_step = self.steps
                events["contact"] = 1.0
            if self.contact_step is not None:
                self.task_phase = "contact"
            if self.intercept_step is None and float(d.min()) < HOSTILE_ENGAGE_RADIUS:
                self.intercept_step = self.steps
                self.task_phase = "intercept"
                events["intercept"] = 1.0
        elif sid == "defend-asset":
            if self.hostile_alive.size:
                live_hostiles = np.where(self.hostile_alive)[0]
                dist_asset = np.linalg.norm(self.hostile_pos[live_hostiles] - self.asset_pos, axis=1)
                breached = live_hostiles[dist_asset < 1.4]
                for idx in breached:
                    self.hostile_alive[idx] = False
                    self.breaches += 1
                    events["breaches"] = events.get("breaches", 0.0) + 1.0
        elif sid == "navigate-to-target":
            if self.intercept_step is None and live_pos.shape[0] > 0:
                d = np.linalg.norm(live_pos - self.target_pos, axis=1)
                if float(d.min()) < 1.5:
                    self.intercept_step = self.steps
                    self.task_phase = "reached"
                    events["reached"] = 1.0
        return events
    def _resolve_obstacle_collisions(self, live_mask: np.ndarray) -> int:
        return self._resolve_obstacle_collisions_from(live_mask, self.pos.copy())

    def _point_inside_obstacle(self, x: float, y: float, k: int, obs: Obstacle) -> bool:
        if self._obs_is_circle[k]:
            r = float(obs.sx) + AGENT_RADIUS
            dx, dy = x - float(obs.cx), y - float(obs.cy)
            return dx * dx + dy * dy < r * r

        hx = float(obs.sx) + AGENT_RADIUS
        hy = float(obs.sy) + AGENT_RADIUS
        return abs(x - float(obs.cx)) < hx and abs(y - float(obs.cy)) < hy

    def _segment_hits_obstacle(
        self,
        sx: float,
        sy: float,
        ex: float,
        ey: float,
        k: int,
        obs: Obstacle,
    ) -> float | None:
        dx, dy = ex - sx, ey - sy
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        if self._point_inside_obstacle(sx, sy, k, obs):
            return None

        if self._obs_is_circle[k]:
            r = float(obs.sx) + AGENT_RADIUS
            ox, oy = sx - float(obs.cx), sy - float(obs.cy)
            a = dx * dx + dy * dy
            b = 2.0 * (ox * dx + oy * dy)
            c = ox * ox + oy * oy - r * r
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return None
            root = math.sqrt(disc)
            t0 = (-b - root) / (2.0 * a)
            t1 = (-b + root) / (2.0 * a)
            if t1 < 0.0 or t0 > 1.0:
                return None
            return max(0.0, t0)

        min_x = float(obs.cx) - float(obs.sx) - AGENT_RADIUS
        max_x = float(obs.cx) + float(obs.sx) + AGENT_RADIUS
        min_y = float(obs.cy) - float(obs.sy) - AGENT_RADIUS
        max_y = float(obs.cy) + float(obs.sy) + AGENT_RADIUS
        t_enter = 0.0
        t_exit = 1.0

        for start, delta, min_v, max_v in ((sx, dx, min_x, max_x), (sy, dy, min_y, max_y)):
            if abs(delta) < 1e-9:
                if start < min_v or start > max_v:
                    return None
                continue
            a = (min_v - start) / delta
            b = (max_v - start) / delta
            if a > b:
                a, b = b, a
            t_enter = max(t_enter, a)
            t_exit = min(t_exit, b)
            if t_enter > t_exit:
                return None

        if t_exit < 0.0 or t_enter > 1.0:
            return None
        return max(0.0, t_enter)

    def _push_out_of_obstacle(self, px: float, py: float, k: int, obs: Obstacle) -> tuple[float, float, bool]:
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
                return cx + dx / d * r, cy + dy / d * r, True
            return px, py, False

        hx = float(obs.sx) + AGENT_RADIUS
        hy = float(obs.sy) + AGENT_RADIUS
        dx, dy = px - cx, py - cy
        if abs(dx) < hx and abs(dy) < hy:
            # push along the axis of least penetration
            pen_x = hx - abs(dx)
            pen_y = hy - abs(dy)
            if pen_x < pen_y:
                px = cx + math.copysign(hx, dx if dx != 0 else 1.0)
            else:
                py = cy + math.copysign(hy, dy if dy != 0 else 1.0)
            return px, py, True
        return px, py, False

    def _resolve_obstacle_collisions_from(self, live_mask: np.ndarray, start_pos: np.ndarray) -> int:
        """Hard block live agents against any obstacle's expanded footprint.

        Each obstacle footprint is grown by AGENT_RADIUS, then any agent inside
        is shoved to the nearest edge (boxes) or onto the inflated circle
        (cylinders). The segment from the previous to proposed position is also
        swept so high-speed agents cannot tunnel through thin walls. Returns
        the number of live agents that needed correction this step — used as a
        collision count for the reward.
        """
        if not self.obstacles or not live_mask.any():
            return 0
        live_idx = np.where(live_mask)[0]
        collided = 0
        for i in live_idx:
            sx, sy = float(start_pos[i, 0]), float(start_pos[i, 1])
            px, py = float(self.pos[i, 0]), float(self.pos[i, 1])
            hit_any = False

            hit_t: float | None = None
            for k, obs in enumerate(self.obstacles):
                t = self._segment_hits_obstacle(sx, sy, px, py, k, obs)
                if t is not None and (hit_t is None or t < hit_t):
                    hit_t = t
            if hit_t is not None:
                stop_t = max(0.0, hit_t - 1e-4)
                px = sx + (px - sx) * stop_t
                py = sy + (py - sy) * stop_t
                hit_any = True

            # Two passes so an agent pushed out of one obstacle can't still sit
            # inside an adjacent one (rare with our layouts but cheap insurance).
            for _ in range(2):
                still_hit = False
                for k, obs in enumerate(self.obstacles):
                    px, py, hit = self._push_out_of_obstacle(px, py, k, obs)
                    still_hit = still_hit or hit
                    hit_any = hit_any or hit
                if not still_hit:
                    break
            if hit_any:
                self.pos[i, 0] = px
                self.pos[i, 1] = py
                collided += 1
        return collided

    def _separation_vector(self, agent_id: int, radius: float = DECONFLICT_RADIUS) -> np.ndarray:
        """Steering vector that keeps live drones from collapsing into one point."""
        if not self.alive[agent_id]:
            return np.zeros(2, dtype=np.float32)
        steer = np.zeros(2, dtype=np.float32)
        p = self.pos[agent_id]
        for other_id in range(self.n):
            if other_id == agent_id or not self.alive[other_id]:
                continue
            delta = p - self.pos[other_id]
            dist = float(np.linalg.norm(delta))
            if dist >= radius:
                continue
            if dist < 1e-5:
                angle = (agent_id * 2.399963229728653 + other_id) % (2.0 * math.pi)
                away = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                dist = 1e-5
            else:
                away = (delta / dist).astype(np.float32)
            steer += away * (((radius - dist) / radius) ** 2)
        return steer

    def _apply_swarm_deconfliction(self, actions: np.ndarray) -> tuple[np.ndarray, int]:
        safe = actions.copy()
        adjusted = 0
        for i in range(self.n):
            if not self.alive[i]:
                continue
            steer = self._separation_vector(i)
            if np.linalg.norm(steer) <= 1e-6:
                continue
            safe[i] += SEPARATION_STEER * steer
            adjusted += 1
        return np.clip(safe, -1.0, 1.0).astype(np.float32), adjusted

    def _resolve_agent_separation(self, live_mask: np.ndarray) -> int:
        live_ids = np.where(live_mask)[0]
        if live_ids.size < 2:
            return 0
        adjusted = 0
        for _ in range(5):
            moved = False
            for a_idx in range(live_ids.size):
                i = int(live_ids[a_idx])
                for b_idx in range(a_idx + 1, live_ids.size):
                    j = int(live_ids[b_idx])
                    delta = self.pos[i] - self.pos[j]
                    dist = float(np.linalg.norm(delta))
                    if dist >= MIN_AGENT_SEPARATION:
                        continue
                    if dist < 1e-5:
                        angle = (i * 2.399963229728653 + j) % (2.0 * math.pi)
                        away = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                        dist = 1e-5
                    else:
                        away = (delta / dist).astype(np.float32)
                    push = 0.5 * (MIN_AGENT_SEPARATION - dist + 1e-2) * away
                    self.pos[i] += push
                    self.pos[j] -= push
                    moved = True
                    adjusted += 1
            if not moved:
                break
            self.pos = np.clip(self.pos, -self.world_half, self.world_half)
        return adjusted

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
        elif sid == "navigate-to-target":
            # single drone starts at the left end of the corridor
            x = rng.uniform(-0.92 * h, -0.75 * h, n)
            y = rng.uniform(-0.3 * h, 0.3 * h, n)
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
        if sid == "navigate-to-target":
            return self.target_pos.copy()
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

    def _angle_spread_score(self, live_pos: np.ndarray, center: np.ndarray) -> float:
        if live_pos.shape[0] < 2:
            return 0.0
        ang = np.arctan2(live_pos[:, 1] - center[1], live_pos[:, 0] - center[0])
        return float(np.clip(np.std(np.unwrap(ang)) / math.pi, 0.0, 1.0))

    def _task_metrics(self, live_pos: np.ndarray) -> dict[str, float | str]:
        sid = self.scenario_id
        steps = max(1, self.steps)
        metrics: dict[str, float | str] = {
            "task_score": self.coverage_fraction(),
            "primary_metric": self.task_profile.primary_metric,
            "primary_value": self.coverage_fraction(),
            "task_phase": self.task_phase,
        }

        if sid == "drone-vs-drone":
            total = max(1, int(self.hostile_alive.size))
            alive = int(self.hostile_alive.sum()) if self.hostile_alive.size else 0
            kill_rate = 1.0 - alive / total
            orbit_score = 0.0
            if live_pos.shape[0] > 0:
                radius = np.linalg.norm(live_pos, axis=1) / self.world_half
                radial = 1.0 - np.clip(np.abs(radius - 0.32) / 0.32, 0.0, 1.0)
                speed = np.linalg.norm(self.vel[self.alive], axis=1)
                slow = 1.0 - np.clip(speed / 1.0, 0.0, 1.0)
                orbit_score = float(np.mean(radial * 0.7 + slow * 0.3))
            task_score = 0.7 * kill_rate + 0.3 * orbit_score
            metrics.update({
                "hostiles_alive": float(alive),
                "hostiles_eliminated": float(total - alive),
                "kill_rate": float(kill_rate),
                "orbit_score": float(orbit_score),
                "task_score": float(task_score),
                "primary_value": float(task_score),
            })
        elif sid == "moving-target-track":
            custody_fraction = self.custody_steps / steps
            lost_track_fraction = self.lost_track_steps / steps
            mean_dist = 1.0
            spread = 0.0
            if live_pos.shape[0] > 0:
                dist = np.linalg.norm(live_pos - self.target_pos, axis=1)
                mean_dist = float(dist.mean() / (2.0 * self.world_half))
                spread = self._angle_spread_score(live_pos, self.target_pos)
            task_score = 0.75 * custody_fraction + 0.25 * spread
            metrics.update({
                "custody_fraction": float(custody_fraction),
                "lost_track_fraction": float(lost_track_fraction),
                "mean_target_distance": float(mean_dist),
                "angle_spread_score": float(spread),
                "task_score": float(task_score),
                "primary_value": float(custody_fraction),
            })
        elif sid == "search-and-interdict":
            contact_made = 1.0 if self.contact_step is not None else 0.0
            intercept_success = 1.0 if self.intercept_step is not None else 0.0
            contact_speed = 0.0 if self.contact_step is None else 1.0 - min(1.0, self.contact_step / self.max_steps)
            intercept_speed = 0.0 if self.intercept_step is None else 1.0 - min(1.0, self.intercept_step / self.max_steps)
            task_score = 0.35 * contact_made + 0.45 * intercept_success + 0.1 * contact_speed + 0.1 * intercept_speed
            metrics.update({
                "contact_made": contact_made,
                "time_to_contact": float(self.max_steps if self.contact_step is None else self.contact_step),
                "intercept_success": intercept_success,
                "time_to_intercept": float(self.max_steps if self.intercept_step is None else self.intercept_step),
                "task_score": float(task_score),
                "primary_value": float(task_score),
            })
        elif sid == "defend-asset":
            total = max(1, int(self.hostile_alive.size) + self.intercepts + self.breaches)
            asset_integrity = max(0.0, 1.0 - self.breaches / total)
            intercept_score = self.intercepts / total
            ring_score = 0.0
            if live_pos.shape[0] > 0:
                radius = np.linalg.norm(live_pos - self.asset_pos, axis=1) / self.world_half
                ring_score = float(np.mean(1.0 - np.clip(np.abs(radius - 0.48) / 0.48, 0.0, 1.0)))
            task_score = 0.55 * asset_integrity + 0.3 * intercept_score + 0.15 * ring_score
            metrics.update({
                "breaches": float(self.breaches),
                "intercepts": float(self.intercepts),
                "asset_integrity": float(asset_integrity),
                "ring_score": float(ring_score),
                "task_score": float(task_score),
                "primary_value": float(asset_integrity),
            })
        elif sid == "navigate-to-target":
            reached = 1.0 if self.intercept_step is not None else 0.0
            dist_norm = 1.0
            if live_pos.shape[0] > 0:
                dist_norm = float(np.linalg.norm(live_pos[0] - self.target_pos)) / (2.0 * self.world_half)
            approach_score = 1.0 - np.clip(dist_norm, 0.0, 1.0)
            task_score = 0.4 * approach_score + 0.6 * reached
            speed_bonus = 0.0 if self.intercept_step is None else 1.0 - min(1.0, self.intercept_step / self.max_steps)
            metrics.update({
                "reached": reached,
                "distance_to_target": float(dist_norm * 2.0 * self.world_half),
                "approach_score": float(approach_score),
                "speed_bonus": float(speed_bonus),
                "task_score": float(task_score),
                "primary_value": float(task_score),
            })
        return metrics

    def _task_reward(self, live_pos: np.ndarray, events: dict[str, float]) -> float:
        sid = self.scenario_id
        weights = self.task_profile.reward_weights
        if live_pos.shape[0] == 0:
            return 0.0
        reward = 0.0
        if sid == "drone-vs-drone":
            if self.hostile_alive.size and self.hostile_alive.any():
                live_hostile = self.hostile_pos[self.hostile_alive]
                dist_to_hostiles = np.linalg.norm(
                    live_pos[:, None, :] - live_hostile[None, :, :],
                    axis=-1,
                )
                nearest_hostile_idx = np.argmin(dist_to_hostiles, axis=1)
                d = dist_to_hostiles[np.arange(live_pos.shape[0]), nearest_hostile_idx]
                mean_d = float(d.mean())
                progress = (self._d2d_prev_dist - mean_d) / (2.0 * self.world_half)
                reward += weights.get("progress", 0.0) * float(np.clip(progress, -1.0, 1.0))
                self._d2d_prev_dist = mean_d
                reward += weights.get("approach", 0.0) * float(
                    (1.0 - np.clip(d / (2.0 * self.world_half), 0.0, 1.0)).mean()
                )

                hostile_d = dist_to_hostiles.min(axis=0)
                reward += weights.get("pressure", 0.0) * float(
                    (1.0 - np.clip(hostile_d / (0.75 * self.world_half), 0.0, 1.0)).mean()
                )
                covered_hostiles = np.count_nonzero(hostile_d < HOSTILE_ENGAGE_RADIUS * 2.5)
                reward += weights.get("swarm", 0.0) * float(covered_hostiles / max(1, live_hostile.shape[0]))

                nearest_hostile = live_hostile[nearest_hostile_idx]
                to_hostile = nearest_hostile - live_pos
                norm = np.maximum(np.linalg.norm(to_hostile, axis=1, keepdims=True), 1e-6)
                closing = (self.vel[self.alive] * (to_hostile / norm)).sum(axis=1)
                reward += weights.get("closing", 0.0) * float(np.clip(closing, -1.0, 1.0).mean())
            reward += weights.get("kill", 0.0) * events.get("kills", 0.0)
            if self.hostile_alive.size and not self.hostile_alive.any():
                radius = np.linalg.norm(live_pos, axis=1) / self.world_half
                orbit = np.mean(1.0 - np.clip(np.abs(radius - 0.32) / 0.32, 0.0, 1.0))
                reward += weights.get("orbit", 0.0) * float(orbit)
        elif sid == "moving-target-track":
            d = np.linalg.norm(live_pos - self.target_pos, axis=1)
            in_custody = float(np.count_nonzero(d < 2.8) / max(1, live_pos.shape[0]))
            spread = self._angle_spread_score(live_pos, self.target_pos)
            reward += weights.get("custody", 0.0) * in_custody
            reward += weights.get("angle_spread", 0.0) * spread
            reward -= weights.get("distance", 0.0) * float(np.mean(np.clip(d / self.world_half, 0.0, 2.0)))
            reward -= weights.get("lost", 0.0) * events.get("lost_track", 0.0)
        elif sid == "search-and-interdict":
            d = np.linalg.norm(live_pos - self.target_pos, axis=1)
            reward += weights.get("contact", 0.0) * events.get("contact", 0.0)
            reward += weights.get("intercept", 0.0) * events.get("intercept", 0.0)
            if self.contact_step is not None:
                reward += weights.get("approach", 0.0) * (1.0 - float(np.clip(d.min() / self.world_half, 0.0, 1.0)))
                reward -= weights.get("delay", 0.0) * min(1.0, (self.steps - self.contact_step) / max(1, self.max_steps))
        elif sid == "defend-asset":
            radius = np.linalg.norm(live_pos - self.asset_pos, axis=1) / self.world_half
            ring = np.mean(1.0 - np.clip(np.abs(radius - 0.48) / 0.48, 0.0, 1.0))
            reward += weights.get("ring", 0.0) * float(ring)
            reward += weights.get("intercept", 0.0) * events.get("kills", 0.0)
            reward -= weights.get("breach", 0.0) * events.get("breaches", 0.0)
        elif sid == "navigate-to-target":
            if live_pos.shape[0] > 0:
                d = float(np.linalg.norm(live_pos[0] - self.target_pos))
                approach_w = weights.get("approach", 0.0)
                # Potential-based shaping: reward Δdist (positive = getting closer).
                # This avoids local minima — the only fixed point is the goal.
                dist_progress = (self._nav_prev_dist - d) / (2.0 * self.world_half)
                reward += approach_w * 3.0 * dist_progress
                self._nav_prev_dist = d
                # Dense approach reward keeps a goal-distance gradient in the value fn.
                reward += approach_w * (1.0 - np.clip(d / (2.0 * self.world_half), 0.0, 1.0))
                # x-progress bonus to break east-west symmetry and push past obstacles.
                x_progress = float(live_pos[0, 0] - self._nav_prev_x) / self.world_half
                self._nav_prev_x = float(live_pos[0, 0])
                if x_progress > 0:
                    reward += approach_w * 0.6 * x_progress
                # Step-time penalty discourages oscillation.
                if self.intercept_step is None:
                    reward -= 0.008
                reward += weights.get("reach", 0.0) * events.get("reached", 0.0)
        else:
            reward += self._objective_reward(live_pos)
        return float(reward)

    def _task_obs(self, agent_idx: int) -> np.ndarray:
        feats = np.zeros(TASK_DIM, dtype=np.float32)
        h = self.world_half
        own = self.pos[agent_idx]
        sid = self.scenario_id

        if sid == "drone-vs-drone":
            live_ids = np.where(self.hostile_alive)[0] if self.hostile_alive.size else np.array([], dtype=np.int64)
            if live_ids.size:
                d = np.linalg.norm(self.hostile_pos[live_ids] - own, axis=1)
                for slot, k in enumerate(np.argsort(d)[:2]):
                    hid = int(live_ids[k])
                    base = slot * 3
                    feats[base:base + 2] = (self.hostile_pos[hid] - own) / h
                    feats[base + 2] = 1.0
                    vbase = 12 + slot * 2
                    feats[vbase:vbase + 2] = np.clip(self.hostile_vel[hid] / MAX_SPEED, -1.0, 1.0)
            total = max(1, int(self.hostile_alive.size))
            feats[6] = float(self.hostile_alive.sum()) / total if self.hostile_alive.size else 0.0
            feats[7:9] = -own / h
            feats[9] = 1.0 if self.task_phase == "orbit" else 0.0
            feats[10] = (np.linalg.norm(own) / h - 0.32)
            feats[11] = np.linalg.norm(self.vel[agent_idx])
        elif sid == "moving-target-track":
            rel = self.target_pos - own
            feats[0:2] = rel / h
            feats[2:4] = np.clip(self.target_vel / MAX_SPEED, -1.0, 1.0)
            dist = np.linalg.norm(rel)
            feats[4] = 1.0 if dist < 2.8 else 0.0
            if self.alive.any():
                team_d = np.linalg.norm(self.pos[self.alive] - self.target_pos, axis=1)
                feats[5] = 1.0 if np.count_nonzero(team_d < 2.8) >= 2 else 0.0
            feats[6] = np.clip((dist - 2.8) / h, -1.0, 1.0)
            desired = 2.0 * math.pi * (agent_idx / max(1, self.n))
            actual = math.atan2(own[1] - self.target_pos[1], own[0] - self.target_pos[0])
            feats[7] = math.sin(actual - desired)
            feats[10] = min(1.0, float(np.linalg.norm(self.target_vel) / MAX_SPEED))
            feats[11] = self.lost_track_steps / max(1, self.steps)
        elif sid == "search-and-interdict":
            phase_map = {"search": 0, "contact": 1, "intercept": 2}
            feats[phase_map.get(self.task_phase, 0)] = 1.0
            contact_known = self.contact_step is not None
            if contact_known:
                feats[3:5] = (self.target_pos - own) / h
                feats[5] = 1.0
                feats[6] = (self.steps - (self.contact_step or self.steps)) / max(1, self.max_steps)
                feats[8] = np.linalg.norm(self.target_pos - own) / h
            else:
                feats[7] = (h - own[1]) / (2.0 * h)
            feats[9] = 1.0 if self.intercept_step is not None else 0.0
        elif sid == "defend-asset":
            feats[0:2] = (self.asset_pos - own) / h
            idx, dist = self._nearest_live_hostile(own)
            if idx is not None:
                feats[2:4] = (self.hostile_pos[idx] - own) / h
                feats[4] = 1.0
                feats[5:7] = np.clip(self.hostile_vel[idx] / MAX_SPEED, -1.0, 1.0)
                asset_dist = np.linalg.norm(self.hostile_pos[idx] - self.asset_pos)
                feats[7] = asset_dist / h
                feats[8] = 1.0 - np.clip(asset_dist / h, 0.0, 1.0)
            feats[9] = np.linalg.norm(own - self.asset_pos) / h - 0.48
            desired = 2.0 * math.pi * (agent_idx / max(1, self.n))
            actual = math.atan2(own[1], own[0])
            feats[10] = math.sin(actual - desired)
            feats[11] = self.breaches / max(1, self._hostile_count())
        elif sid == "navigate-to-target":
            rel = self.target_pos - own
            dist = float(np.linalg.norm(rel))
            feats[0:2] = rel / h                                              # bearing to goal
            feats[2] = np.clip(dist / (2.0 * h), 0.0, 1.0)                   # normalized distance
            feats[3] = 1.0 if self.intercept_step is not None else 0.0        # reached flag
            feats[4] = math.atan2(float(rel[1]), float(rel[0])) / math.pi     # heading angle to goal
        return np.clip(feats, -1.0, 1.0)

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
        self._reset_task_entities()
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
        actions, n_deconflicted = self._apply_swarm_deconfliction(actions)
        self.vel = actions  # record applied command (used in obs)
        start_pos = self.pos.copy()
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
        n_collisions = self._resolve_obstacle_collisions_from(live, start_pos)
        if n_collisions:
            self.pos = np.clip(self.pos, -bound, bound)
        n_separated = 0
        for _ in range(3):
            separated = self._resolve_agent_separation(live)
            extra_collisions = self._resolve_obstacle_collisions_from(live, self.pos.copy())
            n_separated += separated
            if extra_collisions:
                n_collisions += int(extra_collisions)
                self.pos = np.clip(self.pos, -bound, bound)
            if not separated and not extra_collisions:
                break

        self._update_task_entities()

        live_pos = self.pos[live]
        events = self._compute_task_transitions(live_pos)
        new_cells = self._mark_covered()

        coverage_scale = self.task_profile.coverage_weight
        if self.scenario_id == "search-and-interdict" and self.contact_step is not None:
            coverage_scale *= 0.15
        reward = COVERAGE_REWARD * coverage_scale * new_cells

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

            reward += self._task_reward(live_pos, events)

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
            "deconflict_count": int(n_deconflicted + n_separated),
        }
        task_metrics = self._task_metrics(self.pos[self.alive])
        info.update(task_metrics)
        info["task_metrics"] = {
            key: value for key, value in task_metrics.items()
            if isinstance(value, (int, float, np.floating))
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

            out[i, o:o + TASK_DIM] = self._task_obs(i)
            o += TASK_DIM

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
        parts.append(self._global_task_state())
        return np.concatenate(parts).astype(np.float32)

    def _global_task_state(self) -> np.ndarray:
        feats = np.zeros(TASK_DIM, dtype=np.float32)
        h = self.world_half
        if self.scenario_id in {"moving-target-track", "search-and-interdict"}:
            feats[0:2] = self.target_pos / h
            feats[2:4] = np.clip(self.target_vel / MAX_SPEED, -1.0, 1.0)
            feats[4] = 1.0 if self.contact_step is not None else 0.0
            feats[5] = 1.0 if self.intercept_step is not None else 0.0
        elif self.scenario_id in {"drone-vs-drone", "defend-asset"}:
            if self.hostile_alive.size:
                live_ids = np.where(self.hostile_alive)[0]
                feats[0] = float(live_ids.size) / max(1, self.hostile_alive.size)
                if live_ids.size:
                    centroid = self.hostile_pos[live_ids].mean(axis=0)
                    feats[1:3] = centroid / h
            feats[3] = self.breaches / max(1, self._hostile_count())
            feats[4] = self.intercepts / max(1, self._hostile_count())
        elif self.scenario_id == "navigate-to-target":
            feats[0:2] = self.target_pos / h
            feats[2] = 1.0 if self.intercept_step is not None else 0.0
        return np.clip(feats, -1.0, 1.0)

    @property
    def state_dim(self) -> int:
        base = self.n * 2 + self.n * 2 + self.n + self.grid * self.grid
        if self.battlefield is not None:
            base += 4  # 4 normalized P0 scalars appended to global state
        base += TASK_DIM
        return base

    def battlefield_hash(self) -> str:
        """SHA-256 of the battlefield config JSON, first 12 hex chars.

        Logged alongside checkpoints so a training run is always traceable to
        its parameter set (issue #15 acceptance criterion).
        """
        if self.battlefield is None:
            raw = json.dumps(
                {
                    "profile": "garrison",
                    "min_agent_separation": MIN_AGENT_SEPARATION,
                    "deconflict_radius": DECONFLICT_RADIUS,
                },
                sort_keys=True,
            )
            return "garrison-" + hashlib.sha256(raw.encode()).hexdigest()[:12]
        raw = json.dumps(
            {
                "battlefield": asdict(self.battlefield),
                "min_agent_separation": MIN_AGENT_SEPARATION,
                "deconflict_radius": DECONFLICT_RADIUS,
            },
            sort_keys=True,
        )
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
