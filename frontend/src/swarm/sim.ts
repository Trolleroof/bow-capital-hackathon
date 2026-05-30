/**
 * sim.ts — faithful TypeScript port of swarm/env.py (SwarmEnv), inference parts.
 *
 * This replicates SwarmEnv's state, reset, step, and (critically) the 36-dim
 * local observation construction EXACTLY as env.py builds it, so the same
 * trained policy.onnx produces the same coordinated behavior in the browser
 * with no Python in the loop.
 *
 * Observation layout per agent (matches env.py `_obs`):
 *   [ 0: 2]  own position (x,y) normalized by WORLD_HALF -> ~[-1,1]
 *            BattlefieldParams: GPS denial adds Gaussian noise σ = gpsDenialLevel×0.2
 *   [ 2: 4]  own velocity (last applied action), in [-1,1]
 *   [ 4:10]  K=3 nearest LIVE neighbors' relative (dx,dy) / WORLD_HALF,
 *            zero-filled when fewer than K live neighbors exist.
 *            BattlefieldParams: each slot zeroed with probability jamDutyCycle
 *   [10:35]  5x5 local coverage patch centered on the agent, row-major.
 *            1.0 = covered OR out-of-bounds, 0.0 = unexplored
 *   [35]     role flag normalized to [0,1]  (= role / max(1, nRoles-1))
 *
 * BattlefieldParams also wires:
 *   - windSpeed / windDirRad: drift applied in step() after agent command
 *   - attritionInjectRate:    per-step probability of a random agent kill
 *   See battlefieldParams.ts for the full schema.
 */

import type { BattlefieldParams } from '../gym/battlefieldParams'

// ------------------------------------------------------------- constants ---
// These mirror env.py module-level defaults EXACTLY.
export const N_AGENTS = 5
export const K_NEIGHBORS = 3
export const PATCH = 5 // local coverage patch is PATCH x PATCH cells
export const GRID = 20 // world coverage grid is GRID x GRID cells
export const WORLD_HALF = 10.0 // world spans [-WORLD_HALF, WORLD_HALF]
export const ALTITUDE = 2.0 // fixed z (point-mass; z not learned)

export const DT = 0.1
export const MAX_SPEED = 6.0 // world-units / second at full throttle

// roles: all "scout" for Phase 0 -> role index 0 for every agent.
const N_ROLES = 5 // len(ROLES) in env.py

export const OWN_DIM = 4
export const NEIGHBOR_DIM = 2 * K_NEIGHBORS // 6
export const PATCH_DIM = PATCH * PATCH // 25
export const ROLE_DIM = 1
export const OBS_DIM = OWN_DIM + NEIGHBOR_DIM + PATCH_DIM + ROLE_DIM // 36
export const ACT_DIM = 2

// world units per grid cell = (2 * WORLD_HALF) / GRID = 1.0 for defaults
const CELL = (2.0 * WORLD_HALF) / GRID

// --------------------------------------------------------- seeded RNG ---
/**
 * mulberry32 — small deterministic PRNG so reset(seed) is reproducible.
 * (Numbers differ from numpy's PCG64, so absolute spawn positions won't match
 * Python bit-for-bit; the policy is robust to spawn and still reaches the same
 * coverage. The env *dynamics* and obs layout are what must match exactly.)
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Box-Muller transform: generate a standard normal sample using two uniform
 * draws from `rand`.  Used for GPS denial noise in observe().
 */
function sampleNormal(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-10)
  const u2 = rand()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

export class SwarmEnv {
  readonly n: number
  readonly k = K_NEIGHBORS
  readonly patch = PATCH
  readonly grid = GRID
  readonly worldHalf = WORLD_HALF
  readonly cell = CELL
  readonly maxSteps: number

  // Battlefield parameters (P0 knobs wired into dynamics and obs)
  readonly battlefield: BattlefieldParams | null

  // Pre-computed wind drift (world-units/step) from battlefield.weather
  private windX = 0
  private windY = 0

  // state
  pos: Float32Array // n*2, row-major (x,y)
  vel: Float32Array // n*2, last applied action
  alive: boolean[]
  covered: Uint8Array // grid*grid, row-major [cx*grid + cy]
  roles: Int32Array
  steps = 0
  t = 0

  private rand: () => number

  constructor(maxSteps = 400, seed = 0, battlefield: BattlefieldParams | null = null) {
    this.battlefield = battlefield
    this.n = battlefield?.logistics.swarmSize ?? N_AGENTS
    // BattlefieldParams.roe.timeLimitSec maps to maxSteps when provided
    this.maxSteps = battlefield
      ? Math.min(battlefield.roe.timeLimitSec, battlefield.logistics.batteryEnvelopeSec)
      : maxSteps
    this.pos = new Float32Array(this.n * 2)
    this.vel = new Float32Array(this.n * 2)
    this.alive = new Array(this.n).fill(true)
    this.covered = new Uint8Array(this.grid * this.grid)
    this.roles = new Int32Array(this.n)
    this.rand = mulberry32(seed)
    // Pre-compute wind vector (world-units per step = speed * DT)
    if (battlefield) {
      const ws = battlefield.weather.windSpeed
      const wd = battlefield.weather.windDirRad
      this.windX = ws * Math.cos(wd) * DT
      this.windY = ws * Math.sin(wd) * DT
    }
    this.reset(seed)
  }

  // --- map world (x,y) -> integer grid cell (cx,cy), clipped to [0,grid-1] ---
  // env.py: idx = ((p + world_half) / cell).astype(int64), clipped.
  // p + world_half is always >= 0, so int64 truncation == floor here.
  private worldToCell(x: number, y: number): [number, number] {
    let cx = Math.floor((x + this.worldHalf) / this.cell)
    let cy = Math.floor((y + this.worldHalf) / this.cell)
    if (cx < 0) cx = 0
    else if (cx > this.grid - 1) cx = this.grid - 1
    if (cy < 0) cy = 0
    else if (cy > this.grid - 1) cy = this.grid - 1
    return [cx, cy]
  }

  // mark cells under live agents as covered; return # newly covered.
  private markCovered(): number {
    let before = 0
    for (let i = 0; i < this.covered.length; i++) before += this.covered[i]
    for (let i = 0; i < this.n; i++) {
      if (!this.alive[i]) continue
      const [cx, cy] = this.worldToCell(this.pos[i * 2], this.pos[i * 2 + 1])
      this.covered[cx * this.grid + cy] = 1
    }
    let after = 0
    for (let i = 0; i < this.covered.length; i++) after += this.covered[i]
    return after - before
  }

  reset(seed?: number): Float32Array {
    if (seed !== undefined) this.rand = mulberry32(seed)
    // spawn agents in a small cluster near center (env.py: world_half * 0.25)
    const spawn = this.worldHalf * 0.25
    for (let i = 0; i < this.n; i++) {
      // uniform(-spawn, spawn)
      this.pos[i * 2] = (this.rand() * 2 - 1) * spawn
      this.pos[i * 2 + 1] = (this.rand() * 2 - 1) * spawn
      this.vel[i * 2] = 0
      this.vel[i * 2 + 1] = 0
      this.alive[i] = true
      this.roles[i] = 0
    }
    this.covered.fill(0)
    this.steps = 0
    this.t = 0
    this.markCovered()
    return this.observe()
  }

  /**
   * step — integrate point-mass kinematics for one tick.
   * actions: Float32Array length n*2, each in [-1,1] (clipped here).
   * Mirrors env.py.step (without the reward computation, which inference
   * does not need).
   *
   * BattlefieldParams P0 effects applied here:
   *   - attritionInjectRate: each live agent killed with this probability
   *   - windSpeed/windDirRad: drift added to position after agent command
   */
  step(actions: Float32Array): void {
    const bound = this.worldHalf
    const bf = this.battlefield

    // ── Battlefield: random attrition (P0) ──────────────────────────────
    if (bf && bf.logistics.attritionInjectRate > 0) {
      const rate = bf.logistics.attritionInjectRate
      for (let i = 0; i < this.n; i++) {
        if (this.alive[i] && this.rand() < rate) {
          this.kill(i)
        }
      }
    }

    for (let i = 0; i < this.n; i++) {
      // clip action to [-1,1] and record as applied velocity command
      let ax = actions[i * 2]
      let ay = actions[i * 2 + 1]
      if (ax < -1) ax = -1
      else if (ax > 1) ax = 1
      if (ay < -1) ay = -1
      else if (ay > 1) ay = 1
      this.vel[i * 2] = ax
      this.vel[i * 2 + 1] = ay

      if (this.alive[i]) {
        // pos += action * MAX_SPEED * DT + wind_drift, then clip to world bounds
        let px = this.pos[i * 2] + ax * MAX_SPEED * DT + this.windX
        let py = this.pos[i * 2 + 1] + ay * MAX_SPEED * DT + this.windY
        if (px < -bound) px = -bound
        else if (px > bound) px = bound
        if (py < -bound) py = -bound
        else if (py > bound) py = bound
        this.pos[i * 2] = px
        this.pos[i * 2 + 1] = py
      }
    }
    this.markCovered()
    this.steps += 1
    this.t += DT
  }

  /**
   * observe — build the (n, OBS_DIM) local observation, flattened row-major,
   * EXACTLY as env.py `_obs`.
   *
   * BattlefieldParams P0 effects applied here:
   *   - gpsDenialLevel: Gaussian noise σ = level×0.2 on obs[0:2]
   *   - jamDutyCycle:   each neighbor slot zeroed with this probability
   */
  observe(): Float32Array {
    const out = new Float32Array(this.n * OBS_DIM)
    const half = this.worldHalf
    const r = Math.floor(this.patch / 2)

    const bf = this.battlefield
    const gpsSigma = bf ? bf.ew.gpsDenialLevel * 0.2 : 0
    const jamProb  = bf ? bf.ew.jamDutyCycle : 0

    for (let i = 0; i < this.n; i++) {
      const base = i * OBS_DIM
      let o = 0

      // [0:2] own position normalized + GPS denial noise (P0)
      let px = this.pos[i * 2] / half
      let py = this.pos[i * 2 + 1] / half
      if (gpsSigma > 0) {
        px += sampleNormal(this.rand) * gpsSigma
        py += sampleNormal(this.rand) * gpsSigma
      }
      out[base + o] = px
      out[base + o + 1] = py
      o += 2

      // [2:4] own velocity (last applied command)
      out[base + o] = this.vel[i * 2]
      out[base + o + 1] = this.vel[i * 2 + 1]
      o += 2

      // [4:4+2K] K nearest LIVE neighbors' relative (dx,dy)/half, zero-filled.
      // Jamming (P0): each slot independently zeroed with probability jamDutyCycle.
      // env.py: build rel for all live others, sort by distance, take first K.
      const others: number[] = []
      for (let j = 0; j < this.n; j++) {
        if (j !== i && this.alive[j]) others.push(j)
      }
      if (others.length > 0) {
        const dist = others.map((j) => {
          const dx = this.pos[j * 2] - this.pos[i * 2]
          const dy = this.pos[j * 2 + 1] - this.pos[i * 2 + 1]
          return Math.sqrt(dx * dx + dy * dy)
        })
        // argsort ascending (stable, like numpy default for this use)
        const order = others
          .map((_, idx) => idx)
          .sort((a, b) => dist[a] - dist[b])
          .slice(0, this.k)
        for (const idx of order) {
          // Jamming: slot left as zero (already 0 from Float32Array init)
          if (jamProb > 0 && this.rand() < jamProb) {
            o += 2  // slot zeroed — CTDE-safe, matches env.py logic
          } else {
            const j = others[idx]
            out[base + o] = (this.pos[j * 2] - this.pos[i * 2]) / half
            out[base + o + 1] = (this.pos[j * 2 + 1] - this.pos[i * 2 + 1]) / half
            o += 2
          }
        }
      }
      // advance past any unfilled neighbor slots (env.py: o = 4 + 2*k)
      o = OWN_DIM + 2 * this.k

      // [10:35] PATCH x PATCH local coverage patch, row-major. OOB = 1.0.
      const [cx, cy] = this.worldToCell(this.pos[i * 2], this.pos[i * 2 + 1])
      for (let a = 0; a < this.patch; a++) {
        const gx = cx - r + a
        for (let b = 0; b < this.patch; b++) {
          const gy = cy - r + b
          let val = 1.0 // out-of-bounds defaults to covered (1.0)
          if (gx >= 0 && gx < this.grid && gy >= 0 && gy < this.grid) {
            val = this.covered[gx * this.grid + gy] ? 1.0 : 0.0
          }
          out[base + o + a * this.patch + b] = val
        }
      }
      o += this.patch * this.patch

      // [35] role flag normalized to [0,1]
      out[base + o] = this.roles[i] / Math.max(1, N_ROLES - 1)
    }
    return out
  }

  /** Kill an agent: it freezes, stops covering, leaves neighbor sets. */
  kill(agentId: number): void {
    if (agentId < 0 || agentId >= this.n) return
    this.alive[agentId] = false
    this.vel[agentId * 2] = 0
    this.vel[agentId * 2 + 1] = 0
  }

  revive(agentId: number): void {
    if (agentId < 0 || agentId >= this.n) return
    this.alive[agentId] = true
  }

  /** Revive every agent (does not reset coverage or positions). */
  reviveAll(): void {
    for (let i = 0; i < this.n; i++) this.alive[i] = true
  }

  /**
   * Read-only view of the coverage grid as booleans, indexed [cx * grid + cy]
   * (same layout as `covered`). Lets the renderer tint covered ground cells.
   */
  coveredCells(): Uint8Array {
    return this.covered
  }

  /** Fraction of grid cells covered, in [0,1]. */
  coverage(): number {
    let s = 0
    for (let i = 0; i < this.covered.length; i++) s += this.covered[i]
    return s / (this.grid * this.grid)
  }

  /** Number of currently-alive agents. */
  nAlive(): number {
    let c = 0
    for (let i = 0; i < this.n; i++) if (this.alive[i]) c++
    return c
  }
}
