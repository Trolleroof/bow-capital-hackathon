"""3D multi-agent HUNT environment — find an evading target and run it down.

This is a **new** environment (not a variant of ``SwarmEnv``) built for the
``hunt-and-seek`` scenario. The previous coverage env was 2D (velocity in the
XY plane, z frozen at a fixed altitude) and its reward was "paint the floor" —
there was no entity to *find* and no payoff for *catching* anything, which is
why a swarm trained on it never looked like it was hunting. This env fixes both:

  * **True 3D.** Position and velocity are ``(x, y, z)``; the action is a 3D
    velocity command. Obstacles are volumetric pillars of varying height — the
    swarm can fly *between* them and *over* the short ones, so altitude is a real
    tactical degree of freedom.
  * **A slower evading target ("the user").** One entity flees from its nearest
    pursuer (slower than the drones, so a coordinated swarm can corner it) and is
    only *observable* when a drone has line-of-sight within sensor range. When
    nobody sees it, agents share only a decaying last-seen estimate.
  * **Dense find → pursue → capture reward.** A smooth shaping term pulls the
    team toward the target the whole episode (this is the gradient the old sparse
    coverage reward lacked), with a first-contact bonus and a big team capture
    bonus when ≥2 drones box the target inside the capture radius.

MAPPO / CTDE is unchanged: the shared-parameter ``Actor`` consumes only the
local per-agent observation; the centralized ``Critic`` consumes
``global_state()`` at train time only. Actors never see global state and send
zero messages to each other.

Public API matches ``SwarmEnv`` so the trainer, ONNX export, bus, and PyBullet
renderer treat it as a drop-in: ``reset`` / ``step`` / ``global_state`` /
``state_dim`` / ``kill`` / ``revive`` / ``coverage_fraction`` /
``battlefield_hash``, with attributes ``n / obs_dim / act_dim / pos / vel /
alive / covered / roles / t / target_pos``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

try:
    from .obstacles import Obstacle, obstacles_for
    from .task_profiles import get_task_profile
except ImportError:  # direct-script execution
    from obstacles import Obstacle, obstacles_for  # type: ignore
    from task_profiles import get_task_profile  # type: ignore

# ------------------------------------------------------------------ defaults ---
N_AGENTS = 5
K_NEIGHBORS = 3
M_OBSTACLES = 3          # how many nearest obstacles enter each local obs
GRID = 20                # coarse coverage grid (telemetry + light search shaping)

WORLD_HALF = 10.0        # world spans [-WORLD_HALF, WORLD_HALF] in x and y
Z_MIN = 0.6              # floor clearance
Z_MAX = 6.0              # ceiling

DT = 0.1
MAX_SPEED = 4.5          # drone world-units / second at full throttle
TARGET_SPEED_FRAC = 0.38  # target is deliberately SLOWER than the drones (~2.6x)
TARGET_SPEED = MAX_SPEED * TARGET_SPEED_FRAC

AGENT_RADIUS = 0.4       # body half-extent used for collision push-out
OBSTACLE_CLEARANCE = 0.35
SENSOR_RANGE = 7.0       # a drone detects the target within this range + LOS
CAPTURE_RADIUS = 2.3     # a drone "on" the target within this range
CAPTURE_MIN_DRONES = 2   # at least two drones must box the target for a catch
FLEE_RADIUS = 6.0        # target only actively flees when a drone is this close

# ── reward weights ───────────────────────────────────────────────────────────
APPROACH_REWARD = 0.6     # weight on (prev_min_dist - cur_min_dist), the dense pull
FIRST_CONTACT_REWARD = 4.0
PURSUE_REWARD = 1.2       # post-contact closing bonus (scales the approach term)
CAPTURE_REWARD = 25.0     # big team payoff for boxing the target in
LOST_CONTACT_PENALTY = 0.5
CROWD_PENALTY = 0.14
CROWD_RADIUS = 1.7
MIN_AGENT_SEPARATION = 1.5
DECONFLICT_RADIUS = 3.0
SEPARATION_STEER = 1.6
COLLISION_PENALTY = 5.0
BOUNDS_PENALTY = 0.4      # smooth penalty for pushing past the soft boundary
EDGE_SOFT_FRAC = 0.85
SEARCH_REWARD = 0.15      # light pre-contact coverage shaping so they spread to search

ROLES = ("hunter", "hunter", "hunter", "hunter", "hunter")


@dataclass(frozen=True)
class HuntStageConfig:
    target_speed_frac: float
    flee_radius: float
    sensor_range: float
    capture_radius: float
    target_x: tuple[float, float]
    target_y: tuple[float, float]
    target_z: tuple[float, float]
    approach_reward: float
    pursue_reward: float
    capture_reward: float
    first_contact_reward: float
    lost_contact_penalty: float
    search_reward: float
    pressure_reward: float
    contact_keep_reward: float


HUNT_STAGE_CONFIGS: dict[str, HuntStageConfig] = {
    # First teach "close and catch": the target starts near the search front,
    # flees later, and capture radius is forgiving.
    "close": HuntStageConfig(
        target_speed_frac=0.22,
        flee_radius=4.2,
        sensor_range=8.5,
        capture_radius=2.0,
        target_x=(-0.20, 0.35),
        target_y=(-0.45, 0.45),
        target_z=(Z_MIN + 0.3, Z_MAX - 1.2),
        approach_reward=1.15,
        pursue_reward=2.0,
        capture_reward=36.0,
        first_contact_reward=6.0,
        lost_contact_penalty=0.25,
        search_reward=0.08,
        pressure_reward=1.4,
        contact_keep_reward=0.08,
    ),
    # Then keep the same task but restore the far-side spawn and most geometry.
    "slow": HuntStageConfig(
        target_speed_frac=0.30,
        flee_radius=5.0,
        sensor_range=7.8,
        capture_radius=1.7,
        target_x=(0.15, 0.70),
        target_y=(-0.65, 0.65),
        target_z=(Z_MIN + 0.35, Z_MAX - 1.35),
        approach_reward=0.9,
        pursue_reward=1.6,
        capture_reward=31.0,
        first_contact_reward=5.0,
        lost_contact_penalty=0.4,
        search_reward=0.12,
        pressure_reward=1.0,
        contact_keep_reward=0.05,
    ),
    # Final deployment distribution.
    "standard": HuntStageConfig(
        target_speed_frac=TARGET_SPEED_FRAC,
        flee_radius=FLEE_RADIUS,
        sensor_range=SENSOR_RANGE,
        capture_radius=1.35,
        target_x=(0.30, 0.80),
        target_y=(-0.70, 0.70),
        target_z=(Z_MIN + 0.4, Z_MAX - 1.5),
        approach_reward=APPROACH_REWARD,
        pursue_reward=PURSUE_REWARD,
        capture_reward=CAPTURE_REWARD,
        first_contact_reward=FIRST_CONTACT_REWARD,
        lost_contact_penalty=LOST_CONTACT_PENALTY,
        search_reward=SEARCH_REWARD,
        pressure_reward=0.65,
        contact_keep_reward=0.03,
    ),
}

HUNT_CURRICULUM_STAGES = tuple(HUNT_STAGE_CONFIGS)


def obs_dim(k: int = K_NEIGHBORS, m_obstacles: int = M_OBSTACLES) -> int:
    """Local observation vector length for a given K / obstacle count.

    Layout (per agent, LOCAL ONLY):
      own pos (3) + own vel (3)              = 6
      K nearest-neighbor relative pos (3K)
      M nearest obstacles (5 each: dx,dy,dz,r_xy,h_z)
      target block (7): rel dx,dy,dz, dist, visible_now, team_contact, age
      role flag (1)
    """
    return 6 + 3 * k + 5 * m_obstacles + 7 + 1


OWN_DIM = 6
ACT_DIM = 3
OBS_DIM = obs_dim()


class Hunt3DEnv:
    """Vectorized 3D hunt env. See module docstring for the full spec."""

    def __init__(
        self,
        n_agents: int = N_AGENTS,
        k_neighbors: int = K_NEIGHBORS,
        m_obstacles: int = M_OBSTACLES,
        grid: int = GRID,
        world_half: float = WORLD_HALF,
        max_steps: int = 360,
        seed: int | None = None,
        scenario_id: str | None = "hunt-and-seek",
        obstacles: list[Obstacle] | None = None,
        battlefield=None,  # accepted for API parity; only max_steps/n are read
        curriculum_stage: str = "standard",
        pursuit_assist: float = 0.0,
    ) -> None:
        if battlefield is not None:
            n_agents = getattr(battlefield, "n_agents", n_agents)
            max_steps = getattr(battlefield, "max_steps", max_steps)
        self.battlefield = battlefield
        self.scenario_id = scenario_id
        if curriculum_stage not in HUNT_STAGE_CONFIGS:
            known = ", ".join(HUNT_STAGE_CONFIGS)
            raise ValueError(f"unknown hunt curriculum stage '{curriculum_stage}'. known: {known}")
        self.curriculum_stage = curriculum_stage
        self.stage = HUNT_STAGE_CONFIGS[curriculum_stage]
        self.pursuit_assist = float(np.clip(pursuit_assist, 0.0, 1.0))
        self.n = int(n_agents)
        self.k = int(k_neighbors)
        self.m_obstacles = int(m_obstacles)
        self.grid = int(grid)
        self.world_half = float(world_half)
        self.max_steps = int(max_steps)
        self.obs_dim = obs_dim(self.k, self.m_obstacles)
        self.act_dim = ACT_DIM
        self.cell = (2.0 * self.world_half) / self.grid
        self.rng = np.random.default_rng(seed)
        # task profile (primary_metric etc.) consumed by train.py / eval.py
        self.task_profile = get_task_profile(scenario_id)
        # offset where the per-agent target block begins in the local obs vector
        self.target_off = 6 + 3 * self.k + 5 * self.m_obstacles

        # state (filled by reset)
        self.pos = np.zeros((self.n, 3), dtype=np.float32)
        self.vel = np.zeros((self.n, 3), dtype=np.float32)
        self.alive = np.ones(self.n, dtype=bool)
        self.roles = np.zeros(self.n, dtype=np.int64)
        self.covered = np.zeros((self.grid, self.grid), dtype=bool)
        self.t = 0.0
        self.steps = 0

        # target ("the user") — kept as 3D internally; target_pos exposes [x,y,z]
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_vel = np.zeros(3, dtype=np.float32)
        self._target_waypoint = np.zeros(3, dtype=np.float32)

        # detection / contact bookkeeping
        self.contact = False             # target currently visible to any drone
        self.ever_contact = False        # has the team ever seen it this episode
        self.last_seen = np.zeros(3, dtype=np.float32)
        self.last_seen_age = 0           # steps since last seen (capped)
        self.captures = 0
        self.collision_count = 0
        self.obstacle_avoidance_count = 0
        self.contact_step: int | None = None
        self._prev_min_dist = 2.0 * self.world_half

        # obstacles (volumetric: xy footprint + z span)
        if obstacles is None and scenario_id is not None:
            obstacles = obstacles_for(scenario_id)
        self.obstacles: list[Obstacle] = list(obstacles or [])
        if self.obstacles:
            self._obs_centers = np.array([[o.cx, o.cy] for o in self.obstacles], dtype=np.float32)
            self._obs_half = np.array([[o.sx, o.sy] for o in self.obstacles], dtype=np.float32)
            self._obs_zc = np.array([o.z_center for o in self.obstacles], dtype=np.float32)
            self._obs_zh = np.array([o.z_extent for o in self.obstacles], dtype=np.float32)
            self._obs_is_circle = np.array([o.kind == "cylinder" for o in self.obstacles], dtype=bool)
        else:
            self._obs_centers = np.zeros((0, 2), dtype=np.float32)
            self._obs_half = np.zeros((0, 2), dtype=np.float32)
            self._obs_zc = np.zeros(0, dtype=np.float32)
            self._obs_zh = np.zeros(0, dtype=np.float32)
            self._obs_is_circle = np.zeros(0, dtype=bool)

    # ------------------------------------------------------------------ reset ---
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        h = self.world_half
        # spawn the swarm clustered along one edge, looking inward
        base_x = -0.8 * h
        self.pos = np.zeros((self.n, 3), dtype=np.float32)
        self.pos[:, 0] = base_x + self.rng.uniform(-0.8, 0.8, self.n)
        self.pos[:, 1] = self.rng.uniform(-0.55 * h, 0.55 * h, self.n)
        self.pos[:, 2] = self.rng.uniform(Z_MIN + 0.5, Z_MAX - 1.5, self.n)
        self.vel = np.zeros((self.n, 3), dtype=np.float32)
        self.alive = np.ones(self.n, dtype=bool)
        self.roles = np.zeros(self.n, dtype=np.int64)
        self.covered = np.zeros((self.grid, self.grid), dtype=bool)
        self.t = 0.0
        self.steps = 0

        self._spawn_target()
        self.contact = False
        self.ever_contact = False
        self.last_seen = self.target_pos.copy()
        self.last_seen_age = 0
        self.captures = 0
        self.collision_count = 0
        self.obstacle_avoidance_count = 0
        self.contact_step = None
        self._prev_min_dist = self._min_dist_to_target()
        self._mark_covered()
        return self._obs()

    def _spawn_target(self) -> None:
        """Place the target on the far side, away from the swarm spawn edge."""
        h = self.world_half
        self.target_pos = np.array(
            [
                self.rng.uniform(self.stage.target_x[0] * h, self.stage.target_x[1] * h),
                self.rng.uniform(self.stage.target_y[0] * h, self.stage.target_y[1] * h),
                self.rng.uniform(self.stage.target_z[0], self.stage.target_z[1]),
            ],
            dtype=np.float32,
        )
        # nudge out of any obstacle it spawned inside
        self.target_pos = self._push_point_out(self.target_pos)
        self.target_vel = np.zeros(3, dtype=np.float32)
        self._roll_waypoint()

    def _roll_waypoint(self) -> None:
        h = self.world_half
        self._target_waypoint = np.array(
            [
                self.rng.uniform(-0.85 * h, 0.85 * h),
                self.rng.uniform(-0.85 * h, 0.85 * h),
                self.rng.uniform(Z_MIN + 0.4, Z_MAX - 1.0),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------- step ---
    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n, self.act_dim)
        actions = np.clip(actions, -1.0, 1.0)
        if self.pursuit_assist > 0.0:
            expert = self.expert_action()
            actions = np.clip(
                (1.0 - self.pursuit_assist) * actions + self.pursuit_assist * expert,
                -1.0,
                1.0,
            ).astype(np.float32)
        actions, n_deconflicted = self._apply_swarm_deconfliction(actions)
        actions, n_avoidance = self._apply_obstacle_clearance(actions)
        if n_avoidance:
            self.obstacle_avoidance_count += int(n_avoidance)
        self.vel = actions  # record applied command (used in obs/yaw)

        live = self.alive
        start = self.pos.copy()
        self.pos[live] += actions[live] * MAX_SPEED * DT
        self._clip_to_world()
        n_projected = self._enforce_obstacle_clearance(live, OBSTACLE_CLEARANCE)
        if n_projected:
            self.obstacle_avoidance_count += int(n_projected)
        n_collisions = self._resolve_collisions(live, start)
        n_separated = 0
        for _ in range(3):
            separated = self._resolve_agent_separation(live)
            projected = self._enforce_obstacle_clearance(live, OBSTACLE_CLEARANCE)
            extra_collisions = self._resolve_collisions(live, self.pos.copy())
            n_separated += separated
            if projected:
                self.obstacle_avoidance_count += int(projected)
            if extra_collisions:
                n_collisions += int(extra_collisions)
            if not separated and not projected and not extra_collisions:
                break

        self._update_target()

        live_pos = self.pos[live]
        new_cells = self._mark_covered()

        # ── detection (partial observability) ─────────────────────────────
        visible = self._target_visible(live_pos)
        self.contact = visible
        first_contact = visible and not self.ever_contact
        if visible:
            self.ever_contact = True
            self.last_seen = self.target_pos.copy()
            self.last_seen_age = 0
            if self.contact_step is None:
                self.contact_step = self.steps
        else:
            self.last_seen_age = min(self.last_seen_age + 1, 999)

        # ── reward ─────────────────────────────────────────────────────────
        cur_min = self._min_dist_to_target()
        reward = 0.0

        # dense pull toward the target (the gradient the old reward lacked)
        closing = self._prev_min_dist - cur_min
        approach_w = self.stage.approach_reward + (self.stage.pursue_reward if self.ever_contact else 0.0)
        reward += approach_w * float(closing)

        # light search shaping pre-contact so they spread out to find it
        if not self.ever_contact:
            reward += self.stage.search_reward * float(new_cells)
        else:
            ring = max(self.stage.capture_radius * 2.5, 1e-6)
            pressure = float(np.clip((ring - cur_min) / ring, 0.0, 1.0))
            reward += self.stage.pressure_reward * pressure
            if visible:
                reward += self.stage.contact_keep_reward

        if first_contact:
            reward += self.stage.first_contact_reward
        if self.ever_contact and not visible and self.last_seen_age == 1:
            reward -= self.stage.lost_contact_penalty

        # capture: ≥CAPTURE_MIN_DRONES boxing the target in
        n_capturing = self._n_drones_within(live_pos, self.stage.capture_radius)
        captured = visible and n_capturing >= CAPTURE_MIN_DRONES
        if captured:
            reward += self.stage.capture_reward
            self.captures += 1
            # respawn the target elsewhere so the episode yields repeated catches
            self._spawn_target()
            self.contact = False
            self.last_seen = self.target_pos.copy()
            self.last_seen_age = 0
            cur_min = self._min_dist_to_target()

        # crowding
        if live_pos.shape[0] > 1:
            d = np.linalg.norm(live_pos[:, None, :] - live_pos[None, :, :], axis=-1)
            iu = np.triu_indices(live_pos.shape[0], k=1)
            reward -= CROWD_PENALTY * int((d[iu] < CROWD_RADIUS).sum())

        # smooth boundary penalty (xy)
        if live_pos.shape[0] > 0:
            dist_c = np.linalg.norm(live_pos[:, :2], axis=1) / self.world_half
            excess = np.clip(dist_c - EDGE_SOFT_FRAC, 0.0, None)
            reward -= BOUNDS_PENALTY * float((excess ** 2).sum())

        if n_collisions:
            self.collision_count += int(n_collisions)
            reward -= COLLISION_PENALTY * float(n_collisions)

        self._prev_min_dist = cur_min
        self.steps += 1
        self.t += DT
        done = self.steps >= self.max_steps

        rewards = np.where(self.alive, np.float32(reward), np.float32(0.0)).astype(np.float32)
        dones = np.full(self.n, done, dtype=bool)
        info: dict = {
            "new_cells": int(new_cells),
            "coverage": self.coverage_fraction(),
            "n_alive": int(self.alive.sum()),
            "n_collisions": int(n_collisions),
            "collision_count": int(self.collision_count),
            "obstacle_avoidance_count": int(self.obstacle_avoidance_count),
            "deconflict_count": int(n_deconflicted + n_separated),
            "contact": bool(self.contact),
            "captures": int(self.captures),
            "safe_captures": float(self.captures) - 0.03 * float(self.collision_count),
            "min_dist": float(cur_min),
            "min_agent_spacing": float(self._min_agent_spacing()),
            "hunt_stage": self.curriculum_stage,
            "pursuit_assist": self.pursuit_assist,
            "task_metrics": {
                "min_dist": float(cur_min),
                "min_agent_spacing": float(self._min_agent_spacing()),
                "captures": float(self.captures),
                "safe_captures": float(self.captures) - 0.03 * float(self.collision_count),
                "collision_count": float(self.collision_count),
                "obstacle_avoidance_count": float(self.obstacle_avoidance_count),
                "deconflict_count": float(n_deconflicted + n_separated),
                "contact": 1.0 if self.contact else 0.0,
                "capture_pressure": float(
                    np.clip((max(self.stage.capture_radius * 2.5, 1e-6) - cur_min) / max(self.stage.capture_radius * 2.5, 1e-6), 0.0, 1.0)
                ),
            },
        }
        return self._obs(), rewards, dones, info

    # ------------------------------------------------------------- target AI ---
    def _update_target(self) -> None:
        """Slower-than-drones evasion: flee nearest pursuer, else wander."""
        live_pos = self.pos[self.alive]
        if live_pos.shape[0]:
            deltas = self.target_pos[None, :] - live_pos
            dists = np.linalg.norm(deltas, axis=1)
            nearest = int(np.argmin(dists))
            nearest_d = float(dists[nearest])
        else:
            nearest_d = 1e9

        if nearest_d < self.stage.flee_radius:
            flee = self.target_pos - live_pos[nearest]
            norm = np.linalg.norm(flee)
            direction = flee / norm if norm > 1e-6 else self.rng.uniform(-1, 1, 3)
            # bias back toward the open center when near a wall so it doesn't pin
            to_center = -self.target_pos.copy()
            to_center[2] = 0.0
            cdist = np.linalg.norm(self.target_pos[:2])
            if cdist > 0.78 * self.world_half:
                direction = 0.6 * direction + 0.4 * (to_center / (np.linalg.norm(to_center) + 1e-6))
            speed = MAX_SPEED * self.stage.target_speed_frac
        else:
            # wander toward a roaming waypoint at reduced speed
            to_wp = self._target_waypoint - self.target_pos
            if np.linalg.norm(to_wp) < 1.0:
                self._roll_waypoint()
                to_wp = self._target_waypoint - self.target_pos
            direction = to_wp / (np.linalg.norm(to_wp) + 1e-6)
            speed = 0.45 * MAX_SPEED * self.stage.target_speed_frac

        direction = self._avoid_obstacles(self.target_pos, direction)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        self.target_vel = (direction * speed).astype(np.float32)
        nxt = self.target_pos + self.target_vel * DT
        nxt[0] = float(np.clip(nxt[0], -self.world_half, self.world_half))
        nxt[1] = float(np.clip(nxt[1], -self.world_half, self.world_half))
        nxt[2] = float(np.clip(nxt[2], Z_MIN, Z_MAX))
        self.target_pos = self._push_point_out(nxt).astype(np.float32)

    def _avoid_obstacles(self, point: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Add a steering push away from any obstacle whose z-span the point shares."""
        if not self.obstacles:
            return direction
        steer = direction.copy()
        for k in range(len(self.obstacles)):
            if not self._z_overlaps(point[2], k):
                continue
            d2 = point[:2] - self._obs_centers[k]
            reach = float(self._obs_half[k].max()) + 2.4
            dist = float(np.linalg.norm(d2))
            if dist < reach:
                away = d2 / (dist + 1e-6)
                steer[:2] += (reach - dist) / reach * away * 3.0
                top = float(self._obs_zc[k] + self._obs_zh[k])
                if point[2] < top + 0.8 and top + 0.9 < Z_MAX:
                    steer[2] += (reach - dist) / reach * 1.4
        return steer

    # ----------------------------------------------------------- observations ---
    def _obs(self) -> np.ndarray:
        out = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        h = self.world_half
        for i in range(self.n):
            j = 0
            # own pos (xy normalized by world_half, z by Z_MAX) + vel
            out[i, 0] = self.pos[i, 0] / h
            out[i, 1] = self.pos[i, 1] / h
            out[i, 2] = (self.pos[i, 2] - Z_MIN) / (Z_MAX - Z_MIN) * 2.0 - 1.0
            out[i, 3:6] = self.vel[i]
            j = 6
            # K nearest live neighbors (relative position)
            j = self._fill_neighbors(out, i, j)
            # M nearest obstacles
            j = self._fill_obstacles(out, i, j)
            # target block
            j = self._fill_target(out, i, j)
            # role flag
            out[i, j] = float(self.roles[i]) / max(1, len(ROLES) - 1)
        return out

    def _fill_neighbors(self, out: np.ndarray, i: int, j: int) -> int:
        h = self.world_half
        others = [
            (np.linalg.norm(self.pos[m] - self.pos[i]), m)
            for m in range(self.n)
            if m != i and self.alive[m]
        ]
        others.sort(key=lambda t: t[0])
        for s in range(self.k):
            if s < len(others):
                m = others[s][1]
                rel = (self.pos[m] - self.pos[i])
                out[i, j + 0] = rel[0] / h
                out[i, j + 1] = rel[1] / h
                out[i, j + 2] = rel[2] / (Z_MAX - Z_MIN)
            j += 3
        return j

    def _fill_obstacles(self, out: np.ndarray, i: int, j: int) -> int:
        h = self.world_half
        if self.obstacles:
            d = np.linalg.norm(self._obs_centers - self.pos[i, :2], axis=1)
            order = np.argsort(d)[: self.m_obstacles]
        else:
            order = []
        for s in range(self.m_obstacles):
            if s < len(order):
                k = int(order[s])
                out[i, j + 0] = (self._obs_centers[k, 0] - self.pos[i, 0]) / h
                out[i, j + 1] = (self._obs_centers[k, 1] - self.pos[i, 1]) / h
                out[i, j + 2] = (self._obs_zc[k] - self.pos[i, 2]) / (Z_MAX - Z_MIN)
                out[i, j + 3] = float(self._obs_half[k].max()) / h
                out[i, j + 4] = float(self._obs_zh[k]) / (Z_MAX - Z_MIN)
            j += 5
        return j

    def _fill_target(self, out: np.ndarray, i: int, j: int) -> int:
        h = self.world_half
        # personally visible? else fall back to shared last-seen estimate
        seen_now = self._los_clear(self.pos[i]) and (
            np.linalg.norm(self.target_pos - self.pos[i]) < self.stage.sensor_range
        )
        est = self.target_pos if seen_now else self.last_seen
        rel = est - self.pos[i]
        dist = float(np.linalg.norm(rel))
        out[i, j + 0] = rel[0] / h
        out[i, j + 1] = rel[1] / h
        out[i, j + 2] = rel[2] / (Z_MAX - Z_MIN)
        out[i, j + 3] = min(dist / (2.0 * h), 1.0)
        out[i, j + 4] = 1.0 if seen_now else 0.0
        out[i, j + 5] = 1.0 if self.contact else 0.0
        out[i, j + 6] = min(self.last_seen_age / 50.0, 1.0)
        return j + 7

    # --------------------------------------------------------- detection / LOS ---
    def _target_visible(self, live_pos: np.ndarray) -> bool:
        for p in live_pos:
            if np.linalg.norm(self.target_pos - p) < self.stage.sensor_range and self._los_clear(p):
                return True
        return False

    def expert_action(self) -> np.ndarray:
        """Decentralized hunt expert used for BC and residual pursuit assist.

        Before contact, the team flies as a spread search wedge toward the best
        target estimate. After contact, it expands into a ring around the target.
        This avoids both "scatter everywhere" search and the bad visual failure
        mode where every drone collapses onto one pursuit point.
        """
        goal = self.target_pos if self.contact or self.ever_contact else self.last_seen
        out = np.zeros((self.n, self.act_dim), dtype=np.float32)
        live_ids = np.where(self.alive)[0]
        if not live_ids.shape[0]:
            return out
        live_pos = self.pos[live_ids]
        centroid = live_pos.mean(axis=0)
        to_goal = goal - centroid
        to_goal[2] *= 0.65
        norm_goal = float(np.linalg.norm(to_goal))
        forward = to_goal / norm_goal if norm_goal > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        lateral = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
        lat_norm = float(np.linalg.norm(lateral))
        lateral = lateral / lat_norm if lat_norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        center_offset = self.pos - centroid

        if not self.ever_contact:
            slots = np.linspace(-1.0, 1.0, max(1, live_ids.shape[0]), dtype=np.float32)
            for order, i in enumerate(live_ids):
                # Search wedge: enough lateral/vertical lane spacing that drones
                # stay visibly distinct while still pushing toward the estimate.
                lane = float(slots[order])
                desired = centroid + forward * 1.4 + lateral * lane * 1.65
                desired -= forward * (abs(lane) * 0.55)
                desired[2] = float(np.clip(goal[2] + lane * 0.75, Z_MIN + 0.4, Z_MAX - 0.4))
                vec = desired - self.pos[i]
                vec += 0.50 * (goal - self.pos[i])
                vec += 0.18 * center_offset[i]
                vec += 0.45 * self._separation_vector(int(i))
                vec = self._avoid_obstacles(self.pos[i], vec)
                norm = float(np.linalg.norm(vec))
                if norm > 1e-6:
                    out[i] = vec / norm
            return out

        radius = max(self.stage.capture_radius, 1.35)
        for order, i in enumerate(live_ids):
            angle = (2.0 * math.pi * order) / max(1, live_ids.shape[0])
            flank = np.array(
                [math.cos(angle) * radius, math.sin(angle) * radius, 0.0],
                dtype=np.float32,
            )
            desired = goal + flank
            desired[0] = float(np.clip(desired[0], -self.world_half, self.world_half))
            desired[1] = float(np.clip(desired[1], -self.world_half, self.world_half))
            desired[2] = float(np.clip(desired[2], Z_MIN + 0.2, Z_MAX - 0.2))
            vec = desired - self.pos[i]
            vec += 0.20 * (goal - self.pos[i])
            vec += 0.10 * center_offset[i]
            vec += 0.55 * self._separation_vector(int(i))
            vec = self._avoid_obstacles(self.pos[i], vec)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                out[i] = vec / norm
        return out

    def _los_clear(self, src: np.ndarray) -> bool:
        """Line-of-sight from a drone to the target, blocked by tall obstacles."""
        if not self.obstacles:
            return True
        sx, sy = float(src[0]), float(src[1])
        ex, ey = float(self.target_pos[0]), float(self.target_pos[1])
        mid_z = 0.5 * (float(src[2]) + float(self.target_pos[2]))
        for k in range(len(self.obstacles)):
            if not self._z_overlaps(mid_z, k):
                continue  # we (or the target) are above this obstacle — clear
            if self._segment_hits_xy(sx, sy, ex, ey, k):
                return False
        return True

    def _segment_hits_xy(self, sx, sy, ex, ey, k: int) -> bool:
        cx, cy = float(self._obs_centers[k, 0]), float(self._obs_centers[k, 1])
        dx, dy = ex - sx, ey - sy
        if self._obs_is_circle[k]:
            r = float(self._obs_half[k, 0])
            ox, oy = sx - cx, sy - cy
            a = dx * dx + dy * dy
            if a < 1e-9:
                return False
            b = 2.0 * (ox * dx + oy * dy)
            c = ox * ox + oy * oy - r * r
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return False
            root = math.sqrt(disc)
            t0, t1 = (-b - root) / (2 * a), (-b + root) / (2 * a)
            return not (t1 < 0.0 or t0 > 1.0)
        hx, hy = float(self._obs_half[k, 0]), float(self._obs_half[k, 1])
        t_enter, t_exit = 0.0, 1.0
        for start, delta, lo, hi in (
            (sx, dx, cx - hx, cx + hx),
            (sy, dy, cy - hy, cy + hy),
        ):
            if abs(delta) < 1e-9:
                if start < lo or start > hi:
                    return False
                continue
            a = (lo - start) / delta
            b = (hi - start) / delta
            if a > b:
                a, b = b, a
            t_enter = max(t_enter, a)
            t_exit = min(t_exit, b)
            if t_enter > t_exit:
                return False
        return not (t_exit < 0.0 or t_enter > 1.0)

    # --------------------------------------------------------------- collisions ---
    def _z_overlaps(self, z: float, k: int) -> bool:
        lo = float(self._obs_zc[k]) - float(self._obs_zh[k]) - AGENT_RADIUS
        hi = float(self._obs_zc[k]) + float(self._obs_zh[k]) + AGENT_RADIUS
        return lo <= z <= hi

    def _point_inside(self, x: float, y: float, z: float, k: int) -> bool:
        return self._point_inside_with_margin(x, y, z, k, 0.0)

    def _point_inside_with_margin(self, x: float, y: float, z: float, k: int, margin: float) -> bool:
        if not self._z_overlaps(z, k):
            return False  # flying over (or under) the obstacle — no collision
        cx, cy = float(self._obs_centers[k, 0]), float(self._obs_centers[k, 1])
        if self._obs_is_circle[k]:
            r = float(self._obs_half[k, 0]) + AGENT_RADIUS + margin
            return (x - cx) ** 2 + (y - cy) ** 2 < r * r
        hx = float(self._obs_half[k, 0]) + AGENT_RADIUS + margin
        hy = float(self._obs_half[k, 1]) + AGENT_RADIUS + margin
        return abs(x - cx) < hx and abs(y - cy) < hy

    def _separation_vector(self, agent_id: int, radius: float = DECONFLICT_RADIUS) -> np.ndarray:
        """Steering vector that keeps live drones from collapsing into one stack."""
        if not self.alive[agent_id]:
            return np.zeros(3, dtype=np.float32)
        steer = np.zeros(3, dtype=np.float32)
        p = self.pos[agent_id]
        for other_id in range(self.n):
            if other_id == agent_id or not self.alive[other_id]:
                continue
            delta_xy = p[:2] - self.pos[other_id, :2]
            dist = float(np.linalg.norm(delta_xy))
            if dist >= radius:
                continue
            if dist < 1e-5:
                angle = (agent_id * 2.399963229728653 + other_id) % (2.0 * math.pi)
                away_xy = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                dist = 1e-5
            else:
                away_xy = (delta_xy / dist).astype(np.float32)
            strength = ((radius - dist) / radius) ** 2
            steer[:2] += away_xy * strength
            z_gap = float(p[2] - self.pos[other_id, 2])
            if abs(z_gap) < 0.45 and dist < MIN_AGENT_SEPARATION:
                steer[2] += (1.0 if z_gap >= 0.0 else -1.0) * strength * 0.35
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
        for _ in range(6):
            moved = False
            for a_idx in range(live_ids.size):
                i = int(live_ids[a_idx])
                for b_idx in range(a_idx + 1, live_ids.size):
                    j = int(live_ids[b_idx])
                    delta = self.pos[i, :2] - self.pos[j, :2]
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
                    self.pos[i, :2] += push
                    self.pos[j, :2] -= push
                    moved = True
                    adjusted += 1
            if not moved:
                break
            self._clip_to_world()
        return adjusted

    def _apply_obstacle_clearance(self, actions: np.ndarray) -> tuple[np.ndarray, int]:
        if not self.obstacles:
            return actions, 0
        safe = actions.copy()
        n_adjusted = 0
        for i in range(self.n):
            if not self.alive[i]:
                continue
            nxt = self.pos[i] + safe[i] * MAX_SPEED * DT
            nxt[0] = float(np.clip(nxt[0], -self.world_half, self.world_half))
            nxt[1] = float(np.clip(nxt[1], -self.world_half, self.world_half))
            nxt[2] = float(np.clip(nxt[2], Z_MIN, Z_MAX))
            adjusted = False
            for k in range(len(self.obstacles)):
                if self._point_inside_with_margin(float(nxt[0]), float(nxt[1]), float(nxt[2]), k, OBSTACLE_CLEARANCE):
                    nxt[0], nxt[1] = self._push_out_xy_with_margin(
                        float(nxt[0]), float(nxt[1]), k, OBSTACLE_CLEARANCE
                    )
                    adjusted = True
            if adjusted:
                safe[i] = np.clip((nxt - self.pos[i]) / (MAX_SPEED * DT), -1.0, 1.0)
                n_adjusted += 1
        return safe.astype(np.float32), n_adjusted

    def _enforce_obstacle_clearance(self, live_mask: np.ndarray, margin: float) -> int:
        n_adjusted = 0
        for i in range(self.n):
            if not live_mask[i]:
                continue
            x, y, z = float(self.pos[i, 0]), float(self.pos[i, 1]), float(self.pos[i, 2])
            adjusted = False
            for k in range(len(self.obstacles)):
                if self._point_inside_with_margin(x, y, z, k, margin):
                    x, y = self._push_out_xy_with_margin(x, y, k, margin)
                    adjusted = True
            if adjusted:
                self.pos[i, 0] = x
                self.pos[i, 1] = y
                n_adjusted += 1
        if n_adjusted:
            self._clip_to_world()
        return n_adjusted

    def _resolve_collisions(self, live_mask: np.ndarray, start_pos: np.ndarray) -> int:
        n_hits = 0
        for i in range(self.n):
            if not live_mask[i]:
                continue
            x, y, z = float(self.pos[i, 0]), float(self.pos[i, 1]), float(self.pos[i, 2])
            for k in range(len(self.obstacles)):
                if self._point_inside(x, y, z, k):
                    x, y = self._push_out_xy(x, y, k)
                    n_hits += 1
            self.pos[i, 0] = x
            self.pos[i, 1] = y
        self._clip_to_world()
        return n_hits

    def _push_out_xy(self, x: float, y: float, k: int) -> tuple[float, float]:
        return self._push_out_xy_with_margin(x, y, k, 0.0)

    def _push_out_xy_with_margin(self, x: float, y: float, k: int, margin: float) -> tuple[float, float]:
        cx, cy = float(self._obs_centers[k, 0]), float(self._obs_centers[k, 1])
        if self._obs_is_circle[k]:
            r = float(self._obs_half[k, 0]) + AGENT_RADIUS + margin + 1e-3
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy)
            if d < 1e-6:
                dx, dy, d = 1.0, 0.0, 1.0
            return cx + dx / d * r, cy + dy / d * r
        hx = float(self._obs_half[k, 0]) + AGENT_RADIUS + margin + 1e-3
        hy = float(self._obs_half[k, 1]) + AGENT_RADIUS + margin + 1e-3
        dx, dy = x - cx, y - cy
        # push out along the axis of least penetration
        pen_x = hx - abs(dx)
        pen_y = hy - abs(dy)
        if pen_x < pen_y:
            x = cx + math.copysign(hx, dx if dx != 0 else 1.0)
        else:
            y = cy + math.copysign(hy, dy if dy != 0 else 1.0)
        return x, y

    def _push_point_out(self, p: np.ndarray) -> np.ndarray:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        for k in range(len(self.obstacles)):
            if self._point_inside(x, y, z, k):
                x, y = self._push_out_xy(x, y, k)
        return np.array([x, y, z], dtype=np.float32)

    # -------------------------------------------------------------------- misc ---
    def _clip_to_world(self) -> None:
        self.pos[:, 0] = np.clip(self.pos[:, 0], -self.world_half, self.world_half)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -self.world_half, self.world_half)
        self.pos[:, 2] = np.clip(self.pos[:, 2], Z_MIN, Z_MAX)

    def _min_dist_to_target(self) -> float:
        live_pos = self.pos[self.alive]
        if not live_pos.shape[0]:
            return 2.0 * self.world_half
        return float(np.linalg.norm(live_pos - self.target_pos[None, :], axis=1).min())

    def _min_agent_spacing(self) -> float:
        live_pos = self.pos[self.alive]
        if live_pos.shape[0] < 2:
            return 2.0 * self.world_half
        d = np.linalg.norm(live_pos[:, None, :] - live_pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(np.min(d))

    def _n_drones_within(self, live_pos: np.ndarray, radius: float) -> int:
        if not live_pos.shape[0]:
            return 0
        return int((np.linalg.norm(live_pos - self.target_pos[None, :], axis=1) < radius).sum())

    def _world_to_cell(self, p: np.ndarray) -> tuple[int, int]:
        h = self.world_half
        cx = int(np.clip((p[0] + h) / self.cell, 0, self.grid - 1))
        cy = int(np.clip((p[1] + h) / self.cell, 0, self.grid - 1))
        return cx, cy

    def _mark_covered(self) -> int:
        new = 0
        for i in range(self.n):
            if not self.alive[i]:
                continue
            cx, cy = self._world_to_cell(self.pos[i])
            if not self.covered[cx, cy]:
                self.covered[cx, cy] = True
                new += 1
        return new

    # -------------------------------------------------- CTDE centralized state ---
    def global_state(self) -> np.ndarray:
        h = self.world_half
        parts = [
            (self.pos[:, :2] / h).reshape(-1),
            ((self.pos[:, 2:3] - Z_MIN) / (Z_MAX - Z_MIN)).reshape(-1),
            self.vel.reshape(-1),
            self.alive.astype(np.float32),
            np.array([
                self.target_pos[0] / h,
                self.target_pos[1] / h,
                (self.target_pos[2] - Z_MIN) / (Z_MAX - Z_MIN),
                self.target_vel[0] / MAX_SPEED,
                self.target_vel[1] / MAX_SPEED,
                self.target_vel[2] / MAX_SPEED,
                1.0 if self.contact else 0.0,
                1.0 if self.ever_contact else 0.0,
            ], dtype=np.float32),
            self.covered.astype(np.float32).reshape(-1),
            np.array([self.pursuit_assist], dtype=np.float32),
        ]
        return np.concatenate(parts).astype(np.float32)

    @property
    def state_dim(self) -> int:
        # pos: n*2 (xy) + n (z); vel: n*3; alive: n; target block: 8; coverage grid; assist scalar
        return self.n * 2 + self.n + self.n * 3 + self.n + 8 + self.grid * self.grid + 1

    def battlefield_hash(self) -> str:
        stage_tag = f"{self.curriculum_stage}-assist{self.pursuit_assist:.2f}"
        if self.battlefield is None:
            raw = json.dumps(
                {
                    "hunt_stage": self.curriculum_stage,
                    "stage_config": self.stage.__dict__,
                    "collision_penalty": COLLISION_PENALTY,
                    "obstacle_clearance": OBSTACLE_CLEARANCE,
                    "min_agent_separation": MIN_AGENT_SEPARATION,
                    "deconflict_radius": DECONFLICT_RADIUS,
                    "capture_min_drones": CAPTURE_MIN_DRONES,
                    "pursuit_assist": round(self.pursuit_assist, 4),
                },
                sort_keys=True,
            )
            return "hunt-3d-" + hashlib.sha256(raw.encode()).hexdigest()[:12]
        from dataclasses import asdict
        raw = json.dumps(
            {
                "battlefield": asdict(self.battlefield),
                "hunt_stage": self.curriculum_stage,
                "stage_config": asdict(self.stage),
                "collision_penalty": COLLISION_PENALTY,
                "obstacle_clearance": OBSTACLE_CLEARANCE,
                "min_agent_separation": MIN_AGENT_SEPARATION,
                "deconflict_radius": DECONFLICT_RADIUS,
                "capture_min_drones": CAPTURE_MIN_DRONES,
                "pursuit_assist": round(self.pursuit_assist, 4),
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def coverage_fraction(self) -> float:
        return float(self.covered.sum()) / float(self.grid * self.grid)

    def kill(self, agent_id: int) -> None:
        self.alive[agent_id] = False
        self.vel[agent_id] = 0.0

    def revive(self, agent_id: int) -> None:
        self.alive[agent_id] = True


if __name__ == "__main__":
    env = Hunt3DEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (env.n, OBS_DIM), (obs.shape, OBS_DIM)
    info = {}
    for t in range(400):
        a = env.rng.uniform(-1, 1, size=(env.n, env.act_dim)).astype(np.float32)
        obs, r, d, info = env.step(a)
        if t == 200:
            env.kill(0)
    print(
        f"OK obs={obs.shape} state_dim={env.state_dim} act_dim={env.act_dim} "
        f"captures={info['captures']} min_dist={info['min_dist']:.2f} alive={info['n_alive']}"
    )
