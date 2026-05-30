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
  [ 2: 4 ]   own velocity (vx, vy)          last applied action, in [-1, 1]
  [ 4: 4+2K] K nearest-neighbor relative positions (dx, dy) each, normalized;
             zero-filled when fewer than K live neighbors exist
  [ .. .. ]  local coverage patch: a (PATCH x PATCH) grid centered on the agent,
             flattened row-major. Each cell is 1.0 if that world cell is already
             covered (or out of bounds), else 0.0. Encourages moving toward
             unexplored space.
  [ -1 ]     role/goal flag (float): role index normalized to [0, 1]

Dimensions (defaults N=5, K=3, PATCH=5):
  OWN_DIM        = 4                 (pos 2 + vel 2)
  NEIGHBOR_DIM   = 2 * K             (= 6)
  PATCH_DIM      = PATCH * PATCH     (= 25)
  ROLE_DIM       = 1
  OBS_DIM        = OWN_DIM + NEIGHBOR_DIM + PATCH_DIM + ROLE_DIM   (= 36)

`obs_dim(K, patch)` recomputes this for non-default configs. The module-level
constant OBS_DIM is for the defaults so Phase 1/2 can import a fixed shape.

================================================================================
ACTION (per agent)
================================================================================
Continuous 2D velocity command in [-1, 1]^2, integrated as point-mass kinematics:
    pos += action * MAX_SPEED * dt   (then clipped to world bounds)
z is held constant. ACT_DIM = 2.

================================================================================
REWARD (shared / team reward, identical for every live agent)
================================================================================
    + COVERAGE_REWARD  per newly-covered grid cell this step (summed over agents)
    - CROWD_PENALTY    per agent pair closer than CROWD_RADIUS (discourage clumping)
    - BOUNDS_PENALTY   per agent pushing into / sitting on a world edge
Dead agents contribute nothing and receive 0 reward.

================================================================================
ALIVE / KILL
================================================================================
Each agent has an `alive` flag. `kill(agent_id)` freezes the agent (it stops
moving, stops covering cells, and is excluded from neighbor sets) — this drives
the "kill an agent, swarm re-covers the gap with zero comms" money demo.
"""

from __future__ import annotations

import numpy as np

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
BOUNDS_PENALTY = 0.02

ROLES = ("scout", "scout", "scout", "scout", "scout")  # all scouts for Phase 0


def obs_dim(k: int = K_NEIGHBORS, patch: int = PATCH) -> int:
    """Compute the local observation vector length for a given K / patch size."""
    return 4 + 2 * k + patch * patch + 1


# Importable fixed dims for the default config (Phase 1/2 import these).
OWN_DIM = 4
NEIGHBOR_DIM = 2 * K_NEIGHBORS
PATCH_DIM = PATCH * PATCH
ROLE_DIM = 1
OBS_DIM = obs_dim()          # 36 for defaults
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
    ) -> None:
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

        # state (filled by reset)
        self.pos = np.zeros((self.n, 2), dtype=np.float32)
        self.vel = np.zeros((self.n, 2), dtype=np.float32)
        self.alive = np.ones(self.n, dtype=bool)
        self.covered = np.zeros((grid, grid), dtype=bool)
        self.roles = np.zeros(self.n, dtype=np.int64)
        self.t = 0
        self.steps = 0

    # ------------------------------------------------------------------ utils ---
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

    # ------------------------------------------------------------------ reset ---
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset world and return per-agent local obs, shape (n, obs_dim)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # spawn agents in a small cluster near center so they must spread out
        spawn = self.world_half * 0.25
        self.pos = self.rng.uniform(-spawn, spawn, size=(self.n, 2)).astype(np.float32)
        self.vel = np.zeros((self.n, 2), dtype=np.float32)
        self.alive = np.ones(self.n, dtype=bool)
        self.covered = np.zeros((self.grid, self.grid), dtype=bool)
        self.roles = np.zeros(self.n, dtype=np.int64)
        self.t = 0.0
        self.steps = 0
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

        live = self.alive
        self.vel = actions  # record applied command (used in obs)
        # only live agents move
        self.pos[live] += actions[live] * MAX_SPEED * DT
        bound = self.world_half
        at_edge = (np.abs(self.pos) >= bound - 1e-6)
        self.pos = np.clip(self.pos, -bound, bound)

        new_cells = self._mark_covered()

        # --- team reward ---
        reward = COVERAGE_REWARD * new_cells

        # crowding: penalize close live pairs
        if live.sum() > 1:
            lp = self.pos[live]
            d = np.linalg.norm(lp[:, None, :] - lp[None, :, :], axis=-1)
            iu = np.triu_indices(lp.shape[0], k=1)
            reward -= CROWD_PENALTY * int((d[iu] < CROWD_RADIUS).sum())

        # bounds: penalize live agents sitting on an edge
        reward -= BOUNDS_PENALTY * int((at_edge[live].any(axis=1)).sum())

        self.steps += 1
        self.t += DT
        done = self.steps >= self.max_steps

        rewards = np.where(self.alive, np.float32(reward), np.float32(0.0)).astype(
            np.float32
        )
        dones = np.full(self.n, done, dtype=bool)
        info = {
            "new_cells": new_cells,
            "coverage": self.coverage_fraction(),
            "n_alive": int(self.alive.sum()),
        }
        return self._obs(), rewards, dones, info

    # ----------------------------------------------------------- observations ---
    def _obs(self) -> np.ndarray:
        """Build the (n, obs_dim) local observation matrix. See module docstring."""
        out = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        half = self.world_half
        norm_pos = self.pos / half  # ~[-1, 1]

        for i in range(self.n):
            o = 0
            out[i, o:o + 2] = norm_pos[i]; o += 2          # own position
            out[i, o:o + 2] = self.vel[i]; o += 2          # own velocity (cmd)

            # K nearest LIVE neighbors (relative position), zero-filled
            others = [j for j in range(self.n) if j != i and self.alive[j]]
            if others:
                rel = self.pos[others] - self.pos[i]
                dist = np.linalg.norm(rel, axis=1)
                order = np.argsort(dist)[: self.k]
                for n_idx in order:
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

            # role flag normalized to [0, 1]
            out[i, o] = self.roles[i] / max(1, len(ROLES) - 1)
        return out

    def global_state(self) -> np.ndarray:
        """Centralized-critic state (train-only, Phase 1).

        Concatenates all agents' positions, velocities, alive flags, plus the
        flattened global coverage grid. NOT used by the decentralized actor.
        """
        return np.concatenate(
            [
                (self.pos / self.world_half).reshape(-1),
                self.vel.reshape(-1),
                self.alive.astype(np.float32),
                self.covered.astype(np.float32).reshape(-1),
            ]
        ).astype(np.float32)

    @property
    def state_dim(self) -> int:
        return self.n * 2 + self.n * 2 + self.n + self.grid * self.grid

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
