/**
 * sim.ts — faithful TypeScript port of swarm/env.py (SwarmEnv), inference parts.
 *
 * This replicates SwarmEnv's state, reset, step, and (critically) the 64-dim
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
 *   [35:47]  M=3 nearest scenario obstacles: per slot (dx/h, dy/h, sx/h, sy/h)
 *            from obstacle center, zero-filled when fewer than M exist.
 *            Driven by frontend/src/swarm/obstacles.ts (mirrors swarm/obstacles.py).
 *   [47:63]  scenario task features (targets, hostiles, rivals, asset cues)
 *   [63]     role flag normalized to [0,1]  (= role / max(1, nRoles-1))
 *
 * BattlefieldParams also wires:
 *   - windSpeed / windDirRad: drift applied in step() after agent command
 *   - attritionInjectRate:    per-step probability of a random agent kill
 *   See battlefieldParams.ts for the full schema.
 */

import type { BattlefieldParams } from '../gym/battlefieldParams'
import { type Obstacle, obstaclesFor } from './obstacles'

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
// Scenario obstacle slots: nearest-M obstacles, each (dx/h, dy/h, sx/h, sy/h)
export const M_OBSTACLES = 3
export const OBSTACLE_FEATS = 4
export const OBSTACLE_DIM = M_OBSTACLES * OBSTACLE_FEATS // 12
export const TASK_DIM = 16
export const AGENT_RADIUS = 0.4
export const MIN_AGENT_SEPARATION = 1.25
export const DECONFLICT_RADIUS = 2.6
export const SEPARATION_STEER = 1.1
export const OBS_DIM =
  OWN_DIM + NEIGHBOR_DIM + PATCH_DIM + OBSTACLE_DIM + TASK_DIM + ROLE_DIM // 64
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

function pointInsideObstacle(x: number, y: number, obs: Obstacle): boolean {
  if (obs.kind === 'cylinder') {
    const r = obs.sx + AGENT_RADIUS
    const dx = x - obs.cx
    const dy = y - obs.cy
    return dx * dx + dy * dy < r * r
  }
  const hx = obs.sx + AGENT_RADIUS
  const hy = obs.sy + AGENT_RADIUS
  return Math.abs(x - obs.cx) < hx && Math.abs(y - obs.cy) < hy
}

function segmentHitObstacle(
  sx: number,
  sy: number,
  ex: number,
  ey: number,
  obs: Obstacle,
): number | null {
  const dx = ex - sx
  const dy = ey - sy
  if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) return null
  if (pointInsideObstacle(sx, sy, obs)) return null

  if (obs.kind === 'cylinder') {
    const r = obs.sx + AGENT_RADIUS
    const ox = sx - obs.cx
    const oy = sy - obs.cy
    const a = dx * dx + dy * dy
    const b = 2 * (ox * dx + oy * dy)
    const c = ox * ox + oy * oy - r * r
    const disc = b * b - 4 * a * c
    if (disc < 0) return null
    const root = Math.sqrt(disc)
    const t0 = (-b - root) / (2 * a)
    const t1 = (-b + root) / (2 * a)
    if (t1 < 0 || t0 > 1) return null
    return Math.max(0, t0)
  }

  const minX = obs.cx - obs.sx - AGENT_RADIUS
  const maxX = obs.cx + obs.sx + AGENT_RADIUS
  const minY = obs.cy - obs.sy - AGENT_RADIUS
  const maxY = obs.cy + obs.sy + AGENT_RADIUS
  let tEnter = 0
  let tExit = 1

  const applySlab = (start: number, delta: number, min: number, max: number) => {
    if (Math.abs(delta) < 1e-9) return start >= min && start <= max
    let a = (min - start) / delta
    let b = (max - start) / delta
    if (a > b) [a, b] = [b, a]
    tEnter = Math.max(tEnter, a)
    tExit = Math.min(tExit, b)
    return tEnter <= tExit
  }

  if (!applySlab(sx, dx, minX, maxX)) return null
  if (!applySlab(sy, dy, minY, maxY)) return null
  if (tExit < 0 || tEnter > 1) return null
  return Math.max(0, tEnter)
}

function pushOutOfObstacle(px: number, py: number, obs: Obstacle): [number, number, boolean] {
  if (obs.kind === 'cylinder') {
    const r = obs.sx + AGENT_RADIUS
    let dx = px - obs.cx
    let dy = py - obs.cy
    const d2 = dx * dx + dy * dy
    if (d2 < r * r) {
      let d = Math.sqrt(d2)
      if (d < 1e-5) {
        dx = 1
        dy = 0
        d = 1
      }
      return [obs.cx + (dx / d) * r, obs.cy + (dy / d) * r, true]
    }
    return [px, py, false]
  }

  const hx = obs.sx + AGENT_RADIUS
  const hy = obs.sy + AGENT_RADIUS
  const dx = px - obs.cx
  const dy = py - obs.cy
  if (Math.abs(dx) < hx && Math.abs(dy) < hy) {
    const penX = hx - Math.abs(dx)
    const penY = hy - Math.abs(dy)
    if (penX < penY) {
      return [obs.cx + (dx >= 0 ? hx : -hx), py, true]
    }
    return [px, obs.cy + (dy >= 0 ? hy : -hy), true]
  }
  return [px, py, false]
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

  // Per-scenario collidable scenery, mirrored from swarm/obstacles.py
  readonly scenarioId: string | null
  readonly obstacles: Obstacle[]

  // Pre-computed wind drift (world-units/step) from battlefield.weather
  private windX = 0
  private windY = 0

  // state
  pos: Float32Array // n*2, row-major (x,y)
  vel: Float32Array // n*2, last applied action
  alive: boolean[]
  covered: Uint8Array // grid*grid, row-major [cx*grid + cy]
  roles: Int32Array
  hostilePos: Float32Array
  hostileVel: Float32Array
  hostileAlive: boolean[]
  targetPos = new Float32Array(2)
  targetVel = new Float32Array(2)
  rivalPos: Float32Array
  rivalVel: Float32Array
  contestedCells: Int32Array
  contestedOwner: Int8Array
  assetPos = new Float32Array(2)
  taskPhase = 'coverage'
  breaches = 0
  intercepts = 0
  custodySteps = 0
  lostTrackSteps = 0
  contactStep: number | null = null
  interceptStep: number | null = null
  contestedScore = 0
  rivalScore = 0
  steps = 0
  t = 0

  private rand: () => number

  private resolveObstacleMotion(sx: number, sy: number, ex: number, ey: number): [number, number] {
    if (this.obstacles.length === 0) return [ex, ey]

    let hitT = Infinity
    for (const obs of this.obstacles) {
      const t = segmentHitObstacle(sx, sy, ex, ey, obs)
      if (t !== null && t < hitT) hitT = t
    }

    let px = ex
    let py = ey
    if (hitT !== Infinity) {
      const stopT = Math.max(0, hitT - 1e-4)
      px = sx + (ex - sx) * stopT
      py = sy + (ey - sy) * stopT
    }

    for (let pass = 0; pass < 2; pass++) {
      let stillHit = false
      for (const obs of this.obstacles) {
        const [nx, ny, hit] = pushOutOfObstacle(px, py, obs)
        px = nx
        py = ny
        stillHit ||= hit
      }
      if (!stillHit) break
    }

    return [px, py]
  }

  private separationVector(agentIdx: number): [number, number] {
    if (!this.alive[agentIdx]) return [0, 0]
    let sx = 0
    let sy = 0
    const px = this.pos[agentIdx * 2]
    const py = this.pos[agentIdx * 2 + 1]
    for (let otherIdx = 0; otherIdx < this.n; otherIdx++) {
      if (otherIdx === agentIdx || !this.alive[otherIdx]) continue
      let dx = px - this.pos[otherIdx * 2]
      let dy = py - this.pos[otherIdx * 2 + 1]
      let dist = Math.hypot(dx, dy)
      if (dist >= DECONFLICT_RADIUS) continue
      if (dist < 1e-5) {
        const angle = (agentIdx * 2.399963229728653 + otherIdx) % (Math.PI * 2)
        dx = Math.cos(angle)
        dy = Math.sin(angle)
        dist = 1e-5
      } else {
        dx /= dist
        dy /= dist
      }
      const strength = ((DECONFLICT_RADIUS - dist) / DECONFLICT_RADIUS) ** 2
      sx += dx * strength
      sy += dy * strength
    }
    return [sx, sy]
  }

  private applySwarmDeconfliction(actions: Float32Array): [Float32Array, number] {
    const safe = new Float32Array(actions)
    let adjusted = 0
    for (let i = 0; i < this.n; i++) {
      if (!this.alive[i]) continue
      const [sx, sy] = this.separationVector(i)
      if (Math.hypot(sx, sy) <= 1e-6) continue
      safe[i * 2] = Math.max(-1, Math.min(1, safe[i * 2] + sx * SEPARATION_STEER))
      safe[i * 2 + 1] = Math.max(-1, Math.min(1, safe[i * 2 + 1] + sy * SEPARATION_STEER))
      adjusted++
    }
    return [safe, adjusted]
  }

  private resolveAgentSeparation(): number {
    const liveIds: number[] = []
    for (let i = 0; i < this.n; i++) if (this.alive[i]) liveIds.push(i)
    if (liveIds.length < 2) return 0
    let adjusted = 0
    for (let pass = 0; pass < 5; pass++) {
      let moved = false
      for (let a = 0; a < liveIds.length; a++) {
        const i = liveIds[a]
        for (let b = a + 1; b < liveIds.length; b++) {
          const j = liveIds[b]
          let dx = this.pos[i * 2] - this.pos[j * 2]
          let dy = this.pos[i * 2 + 1] - this.pos[j * 2 + 1]
          let dist = Math.hypot(dx, dy)
          if (dist >= MIN_AGENT_SEPARATION) continue
          if (dist < 1e-5) {
            const angle = (i * 2.399963229728653 + j) % (Math.PI * 2)
            dx = Math.cos(angle)
            dy = Math.sin(angle)
            dist = 1e-5
          } else {
            dx /= dist
            dy /= dist
          }
          const push = 0.5 * (MIN_AGENT_SEPARATION - dist + 1e-2)
          this.pos[i * 2] = Math.max(-this.worldHalf, Math.min(this.worldHalf, this.pos[i * 2] + dx * push))
          this.pos[i * 2 + 1] = Math.max(-this.worldHalf, Math.min(this.worldHalf, this.pos[i * 2 + 1] + dy * push))
          this.pos[j * 2] = Math.max(-this.worldHalf, Math.min(this.worldHalf, this.pos[j * 2] - dx * push))
          this.pos[j * 2 + 1] = Math.max(-this.worldHalf, Math.min(this.worldHalf, this.pos[j * 2 + 1] - dy * push))
          adjusted++
          moved = true
        }
      }
      if (!moved) break
    }
    return adjusted
  }

  private resolveObstacleOverlaps(): number {
    if (this.obstacles.length === 0) return 0
    let adjusted = 0
    for (let i = 0; i < this.n; i++) {
      if (!this.alive[i]) continue
      let px = this.pos[i * 2]
      let py = this.pos[i * 2 + 1]
      let agentAdjusted = false
      for (let pass = 0; pass < 2; pass++) {
        let hitAny = false
        for (const obs of this.obstacles) {
          const [nx, ny, hit] = pushOutOfObstacle(px, py, obs)
          px = nx
          py = ny
          hitAny ||= hit
          agentAdjusted ||= hit
        }
        if (!hitAny) break
      }
      this.pos[i * 2] = Math.max(-this.worldHalf, Math.min(this.worldHalf, px))
      this.pos[i * 2 + 1] = Math.max(-this.worldHalf, Math.min(this.worldHalf, py))
      if (agentAdjusted) adjusted++
    }
    return adjusted
  }

  constructor(
    maxSteps = 400,
    seed = 0,
    battlefield: BattlefieldParams | null = null,
    scenarioId: string | null = null,
  ) {
    this.battlefield = battlefield
    this.scenarioId = scenarioId
    this.obstacles = obstaclesFor(scenarioId)
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
    this.hostilePos = new Float32Array(0)
    this.hostileVel = new Float32Array(0)
    this.hostileAlive = []
    this.rivalPos = new Float32Array(0)
    this.rivalVel = new Float32Array(0)
    this.contestedCells = new Int32Array(0)
    this.contestedOwner = new Int8Array(0)
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

  private hostileCount(): number {
    if (this.scenarioId !== 'drone-vs-drone' && this.scenarioId !== 'defend-asset') return 0
    return Math.max(1, this.battlefield?.threat.hostileUasCount ?? (this.scenarioId === 'drone-vs-drone' ? 3 : 5))
  }

  private resetTaskEntities(): void {
    const h = this.worldHalf
    this.taskPhase =
      this.scenarioId === 'drone-vs-drone' ? 'engage'
        : this.scenarioId === 'search-and-interdict' ? 'search'
          : this.scenarioId === 'defend-asset' ? 'defend'
            : this.scenarioId === 'moving-target-track' ? 'track'
              : 'coverage'
    this.hostilePos = new Float32Array(0)
    this.hostileVel = new Float32Array(0)
    this.hostileAlive = []
    this.rivalPos = new Float32Array(0)
    this.rivalVel = new Float32Array(0)
    this.contestedCells = new Int32Array(0)
    this.contestedOwner = new Int8Array(0)
    this.assetPos[0] = 0
    this.assetPos[1] = 0
    this.targetPos[0] = 0.45 * h
    this.targetPos[1] = 0.35 * h
    this.targetVel[0] = 0
    this.targetVel[1] = 0
    this.breaches = 0
    this.intercepts = 0
    this.custodySteps = 0
    this.lostTrackSteps = 0
    this.contactStep = null
    this.interceptStep = null
    this.contestedScore = 0
    this.rivalScore = 0

    if (this.scenarioId === 'drone-vs-drone') {
      const count = this.hostileCount()
      this.hostilePos = new Float32Array(count * 2)
      this.hostileVel = new Float32Array(count * 2)
      this.hostileAlive = new Array(count).fill(true)
      for (let i = 0; i < count; i++) {
        this.hostilePos[i * 2] = (0.35 + this.rand() * 0.5) * h
        this.hostilePos[i * 2 + 1] = (this.rand() * 1.3 - 0.65) * h
      }
    } else if (this.scenarioId === 'moving-target-track') {
      const speed = Math.max(0.15, this.battlefield?.threat.movingTargetSpeed ?? 0.6)
      this.targetPos[0] = -0.45 * h
      this.targetPos[1] = -0.15 * h
      this.targetVel[0] = speed
      this.targetVel[1] = speed * 0.45
    } else if (this.scenarioId === 'search-and-interdict') {
      this.targetPos[0] = 0.35 * h
      this.targetPos[1] = 0.45 * h
      this.targetVel[0] = -0.18
      this.targetVel[1] = -0.08
    } else if (this.scenarioId === 'defend-asset') {
      const count = this.hostileCount()
      this.hostilePos = new Float32Array(count * 2)
      this.hostileVel = new Float32Array(count * 2)
      this.hostileAlive = new Array(count).fill(true)
      for (let i = 0; i < count; i++) {
        const a = (i / count) * Math.PI * 2
        const x = Math.cos(a) * 0.95 * h
        const y = Math.sin(a) * 0.95 * h
        this.hostilePos[i * 2] = x
        this.hostilePos[i * 2 + 1] = y
        const d = Math.max(1e-6, Math.hypot(x, y))
        this.hostileVel[i * 2] = (-x / d) * 0.55
        this.hostileVel[i * 2 + 1] = (-y / d) * 0.55
      }
    } else if (this.scenarioId === 'navigate-to-target') {
      this.targetPos[0] = 0.85 * h
      this.targetPos[1] = 0
      this.targetVel[0] = 0
      this.targetVel[1] = 0
    }
  }

  private spawnAgents(): void {
    const h = this.worldHalf
    for (let i = 0; i < this.n; i++) {
      if (this.scenarioId === 'drone-vs-drone') {
        this.pos[i * 2] = -(0.6 + this.rand() * 0.32) * h
        this.pos[i * 2 + 1] = (this.rand() * 1.4 - 0.7) * h
      } else if (this.scenarioId === 'search-and-interdict') {
        this.pos[i * 2] = (this.rand() * 1.7 - 0.85) * h
        this.pos[i * 2 + 1] = -(0.6 + this.rand() * 0.32) * h
      } else if (this.scenarioId === 'defend-asset') {
        const a = this.rand() * Math.PI * 2
        const r = (0.42 + this.rand() * 0.13) * h
        this.pos[i * 2] = Math.cos(a) * r
        this.pos[i * 2 + 1] = Math.sin(a) * r
      } else if (this.scenarioId === 'navigate-to-target') {
        this.pos[i * 2] = -(0.75 + this.rand() * 0.17) * h
        this.pos[i * 2 + 1] = (this.rand() * 0.6 - 0.3) * h
      } else {
        this.pos[i * 2] = (this.rand() * 1.6 - 0.8) * h
        this.pos[i * 2 + 1] = (this.rand() * 1.6 - 0.8) * h
      }
      this.vel[i * 2] = 0
      this.vel[i * 2 + 1] = 0
      this.alive[i] = true
      this.roles[i] = 0
    }
  }

  reset(seed?: number): Float32Array {
    if (seed !== undefined) this.rand = mulberry32(seed)
    this.spawnAgents()
    this.covered.fill(0)
    this.steps = 0
    this.t = 0
    this.resetTaskEntities()
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

    const [safeActions] = this.applySwarmDeconfliction(actions)

    for (let i = 0; i < this.n; i++) {
      // clip action to [-1,1] and record as applied velocity command
      let ax = safeActions[i * 2]
      let ay = safeActions[i * 2 + 1]
      if (ax < -1) ax = -1
      else if (ax > 1) ax = 1
      if (ay < -1) ay = -1
      else if (ay > 1) ay = 1
      this.vel[i * 2] = ax
      this.vel[i * 2 + 1] = ay

      if (this.alive[i]) {
        const startX = this.pos[i * 2]
        const startY = this.pos[i * 2 + 1]
        // pos += action * MAX_SPEED * DT + wind_drift, then clip to world bounds
        let px = startX + ax * MAX_SPEED * DT + this.windX
        let py = startY + ay * MAX_SPEED * DT + this.windY
        if (px < -bound) px = -bound
        else if (px > bound) px = bound
        if (py < -bound) py = -bound
        else if (py > bound) py = bound
        // Swept collision blocks fast agents from tunneling through thin walls.
        if (this.obstacles.length > 0) {
          const resolved = this.resolveObstacleMotion(startX, startY, px, py)
          px = resolved[0]
          py = resolved[1]
          if (px < -bound) px = -bound
          else if (px > bound) px = bound
          if (py < -bound) py = -bound
          else if (py > bound) py = bound
        }
        this.pos[i * 2] = px
        this.pos[i * 2 + 1] = py
      }
    }
    for (let pass = 0; pass < 3; pass++) {
      const separated = this.resolveAgentSeparation()
      const obstacleAdjusted = this.resolveObstacleOverlaps()
      if (!separated && !obstacleAdjusted) break
    }
    this.updateTaskEntities()
    this.computeTaskTransitions()
    this.markCovered()
    this.steps += 1
    this.t += DT
  }

  private updateTaskEntities(): void {
    const h = this.worldHalf
    if (this.scenarioId === 'drone-vs-drone') {
      const cx = 0.45 * h
      for (let i = 0; i < this.hostileAlive.length; i++) {
        if (!this.hostileAlive[i]) continue
        const r = 0.16 * h + i * 0.08 * h
        const a = this.t * 0.45 + i * 2.1
        const nx = cx + Math.cos(a) * r
        const ny = Math.sin(a) * r
        this.hostileVel[i * 2] = (nx - this.hostilePos[i * 2]) / DT
        this.hostileVel[i * 2 + 1] = (ny - this.hostilePos[i * 2 + 1]) / DT
        this.hostilePos[i * 2] = nx
        this.hostilePos[i * 2 + 1] = ny
      }
    } else if (this.scenarioId === 'moving-target-track') {
      const speed = Math.max(0.15, this.battlefield?.threat.movingTargetSpeed ?? 0.6)
      const nx = 0.62 * h * Math.sin(this.t * 0.20 * speed)
      const ny = 0.45 * h * Math.sin(this.t * 0.37 * speed + 0.7)
      this.targetVel[0] = (nx - this.targetPos[0]) / DT
      this.targetVel[1] = (ny - this.targetPos[1]) / DT
      this.targetPos[0] = nx
      this.targetPos[1] = ny
    } else if (this.scenarioId === 'search-and-interdict') {
      const nx = 0.42 * h * Math.sin(this.t * 0.16) + 0.18 * h
      const ny = 0.42 * h * Math.cos(this.t * 0.23)
      this.targetVel[0] = (nx - this.targetPos[0]) / DT
      this.targetVel[1] = (ny - this.targetPos[1]) / DT
      this.targetPos[0] = nx
      this.targetPos[1] = ny
    } else if (this.scenarioId === 'defend-asset') {
      for (let i = 0; i < this.hostileAlive.length; i++) {
        if (!this.hostileAlive[i]) continue
        this.hostilePos[i * 2] += this.hostileVel[i * 2] * DT
        this.hostilePos[i * 2 + 1] += this.hostileVel[i * 2 + 1] * DT
      }
    }
  }

  private computeTaskTransitions(): void {
    const live: number[] = []
    for (let i = 0; i < this.n; i++) if (this.alive[i]) live.push(i)
    if (live.length === 0) return
    if (this.scenarioId === 'drone-vs-drone' || this.scenarioId === 'defend-asset') {
      for (let hidx = 0; hidx < this.hostileAlive.length; hidx++) {
        if (!this.hostileAlive[hidx]) continue
        for (const i of live) {
          const d = Math.hypot(
            this.pos[i * 2] - this.hostilePos[hidx * 2],
            this.pos[i * 2 + 1] - this.hostilePos[hidx * 2 + 1],
          )
          if (d < 2.0) {
            this.hostileAlive[hidx] = false
            this.intercepts += 1
            break
          }
        }
      }
      if (this.scenarioId === 'drone-vs-drone') {
        this.taskPhase = this.hostileAlive.some(Boolean) ? 'engage' : 'orbit'
      }
    }
    if (this.scenarioId === 'moving-target-track') {
      let inCustody = 0
      for (const i of live) {
        if (Math.hypot(this.pos[i * 2] - this.targetPos[0], this.pos[i * 2 + 1] - this.targetPos[1]) < 2.8) {
          inCustody++
        }
      }
      if (inCustody >= Math.max(1, Math.min(2, live.length))) this.custodySteps++
      else this.lostTrackSteps++
    } else if (this.scenarioId === 'search-and-interdict') {
      let nearest = Infinity
      for (const i of live) {
        nearest = Math.min(nearest, Math.hypot(this.pos[i * 2] - this.targetPos[0], this.pos[i * 2 + 1] - this.targetPos[1]))
      }
      if (this.contactStep === null && nearest < 4.2) this.contactStep = this.steps
      if (this.contactStep !== null) this.taskPhase = 'contact'
      if (this.interceptStep === null && nearest < 2.0) {
        this.interceptStep = this.steps
        this.taskPhase = 'intercept'
      }
    } else if (this.scenarioId === 'defend-asset') {
      for (let hidx = 0; hidx < this.hostileAlive.length; hidx++) {
        if (!this.hostileAlive[hidx]) continue
        const d = Math.hypot(this.hostilePos[hidx * 2] - this.assetPos[0], this.hostilePos[hidx * 2 + 1] - this.assetPos[1])
        if (d < 1.4) {
          this.hostileAlive[hidx] = false
          this.breaches++
        }
      }
    } else if (this.scenarioId === 'navigate-to-target') {
      if (this.interceptStep === null) {
        let nearest = Infinity
        for (const i of live) {
          nearest = Math.min(nearest, Math.hypot(this.pos[i * 2] - this.targetPos[0], this.pos[i * 2 + 1] - this.targetPos[1]))
        }
        if (nearest < 1.5) {
          this.interceptStep = this.steps
          this.taskPhase = 'reached'
        }
      }
    }
  }

  private taskObs(agentIdx: number): Float32Array {
    const feats = new Float32Array(TASK_DIM)
    const h = this.worldHalf
    const ox = this.pos[agentIdx * 2]
    const oy = this.pos[agentIdx * 2 + 1]
    if (this.scenarioId === 'drone-vs-drone') {
      const live = this.hostileAlive
        .map((alive, idx) => ({ alive, idx }))
        .filter((x) => x.alive)
        .sort((a, b) => {
          const ad = Math.hypot(this.hostilePos[a.idx * 2] - ox, this.hostilePos[a.idx * 2 + 1] - oy)
          const bd = Math.hypot(this.hostilePos[b.idx * 2] - ox, this.hostilePos[b.idx * 2 + 1] - oy)
          return ad - bd
        })
        .slice(0, 2)
      for (let slot = 0; slot < live.length; slot++) {
        const idx = live[slot].idx
        const base = slot * 3
        feats[base] = (this.hostilePos[idx * 2] - ox) / h
        feats[base + 1] = (this.hostilePos[idx * 2 + 1] - oy) / h
        feats[base + 2] = 1
        // Hostile velocity for the two nearest threats (slots 12-15). Must match
        // env.py _task_obs so the ONNX policy sees the obs it was trained on.
        const vbase = 12 + (base / 3) * 2
        feats[vbase] = Math.max(-1, Math.min(1, this.hostileVel[idx * 2] / MAX_SPEED))
        feats[vbase + 1] = Math.max(-1, Math.min(1, this.hostileVel[idx * 2 + 1] / MAX_SPEED))
      }
      const total = Math.max(1, this.hostileAlive.length)
      feats[6] = this.hostileAlive.filter(Boolean).length / total
      feats[7] = -ox / h
      feats[8] = -oy / h
      feats[9] = this.taskPhase === 'orbit' ? 1 : 0
      feats[10] = Math.hypot(ox, oy) / h - 0.32
      feats[11] = Math.hypot(this.vel[agentIdx * 2], this.vel[agentIdx * 2 + 1])
    } else if (this.scenarioId === 'moving-target-track') {
      const dx = this.targetPos[0] - ox
      const dy = this.targetPos[1] - oy
      const dist = Math.hypot(dx, dy)
      feats[0] = dx / h
      feats[1] = dy / h
      feats[2] = Math.max(-1, Math.min(1, this.targetVel[0] / MAX_SPEED))
      feats[3] = Math.max(-1, Math.min(1, this.targetVel[1] / MAX_SPEED))
      feats[4] = dist < 2.8 ? 1 : 0
      let team = 0
      for (let i = 0; i < this.n; i++) {
        if (this.alive[i] && Math.hypot(this.pos[i * 2] - this.targetPos[0], this.pos[i * 2 + 1] - this.targetPos[1]) < 2.8) team++
      }
      feats[5] = team >= 2 ? 1 : 0
      feats[6] = Math.max(-1, Math.min(1, (dist - 2.8) / h))
      const desired = (agentIdx / Math.max(1, this.n)) * Math.PI * 2
      const actual = Math.atan2(oy - this.targetPos[1], ox - this.targetPos[0])
      feats[7] = Math.sin(actual - desired)
      feats[10] = Math.min(1, Math.hypot(this.targetVel[0], this.targetVel[1]) / MAX_SPEED)
      feats[11] = this.lostTrackSteps / Math.max(1, this.steps)
    } else if (this.scenarioId === 'search-and-interdict') {
      const phase = this.taskPhase === 'intercept' ? 2 : this.taskPhase === 'contact' ? 1 : 0
      feats[phase] = 1
      if (this.contactStep !== null) {
        feats[3] = (this.targetPos[0] - ox) / h
        feats[4] = (this.targetPos[1] - oy) / h
        feats[5] = 1
        feats[6] = (this.steps - this.contactStep) / Math.max(1, this.maxSteps)
        feats[8] = Math.hypot(this.targetPos[0] - ox, this.targetPos[1] - oy) / h
      } else {
        feats[7] = (h - oy) / (2 * h)
      }
      feats[9] = this.interceptStep !== null ? 1 : 0
    } else if (this.scenarioId === 'defend-asset') {
      feats[0] = (this.assetPos[0] - ox) / h
      feats[1] = (this.assetPos[1] - oy) / h
      let nearest = -1
      let nearestDist = Infinity
      for (let i = 0; i < this.hostileAlive.length; i++) {
        if (!this.hostileAlive[i]) continue
        const d = Math.hypot(this.hostilePos[i * 2] - ox, this.hostilePos[i * 2 + 1] - oy)
        if (d < nearestDist) {
          nearest = i
          nearestDist = d
        }
      }
      if (nearest >= 0) {
        feats[2] = (this.hostilePos[nearest * 2] - ox) / h
        feats[3] = (this.hostilePos[nearest * 2 + 1] - oy) / h
        feats[4] = 1
        feats[5] = Math.max(-1, Math.min(1, this.hostileVel[nearest * 2] / MAX_SPEED))
        feats[6] = Math.max(-1, Math.min(1, this.hostileVel[nearest * 2 + 1] / MAX_SPEED))
        const assetDist = Math.hypot(this.hostilePos[nearest * 2], this.hostilePos[nearest * 2 + 1])
        feats[7] = assetDist / h
        feats[8] = 1 - Math.max(0, Math.min(1, assetDist / h))
      }
      feats[9] = Math.hypot(ox, oy) / h - 0.48
      feats[10] = Math.sin(Math.atan2(oy, ox) - (agentIdx / Math.max(1, this.n)) * Math.PI * 2)
      feats[11] = this.breaches / Math.max(1, this.hostileCount())
    } else if (this.scenarioId === 'navigate-to-target') {
      const dx = this.targetPos[0] - ox
      const dy = this.targetPos[1] - oy
      const dist = Math.hypot(dx, dy)
      feats[0] = dx / h
      feats[1] = dy / h
      feats[2] = Math.max(0, Math.min(1, dist / (2 * h)))
      feats[3] = this.interceptStep !== null ? 1 : 0
      feats[4] = Math.atan2(dy, dx) / Math.PI
    }
    for (let i = 0; i < feats.length; i++) {
      feats[i] = Math.max(-1, Math.min(1, feats[i]))
    }
    return feats
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

      // nearest M_OBSTACLES scenario obstacles: (dx/h, dy/h, sx/h, sy/h) each.
      // Empty slots stay zero (already initialised). Mirrors env.py _obs.
      if (this.obstacles.length > 0) {
        const obsList = this.obstacles
        const dists = obsList.map((obs) => {
          const dx = obs.cx - this.pos[i * 2]
          const dy = obs.cy - this.pos[i * 2 + 1]
          return Math.sqrt(dx * dx + dy * dy)
        })
        const order = obsList
          .map((_, idx) => idx)
          .sort((a, b) => dists[a] - dists[b])
          .slice(0, M_OBSTACLES)
        for (let slot = 0; slot < order.length; slot++) {
          const obs = obsList[order[slot]]
          const slotBase = base + o + slot * OBSTACLE_FEATS
          out[slotBase + 0] = (obs.cx - this.pos[i * 2]) / half
          out[slotBase + 1] = (obs.cy - this.pos[i * 2 + 1]) / half
          out[slotBase + 2] = obs.sx / half
          out[slotBase + 3] = obs.sy / half
        }
      }
      o += OBSTACLE_DIM

      const task = this.taskObs(i)
      out.set(task, base + o)
      o += TASK_DIM

      // [final] role flag normalized to [0,1]
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
