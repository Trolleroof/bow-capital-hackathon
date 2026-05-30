/**
 * GymScenarioStage.tsx
 *
 * Owns the full gym stage area:
 *   • Scripted scene animation (unchanged)
 *   • Issue #25 — BattlefieldParamsPanel (P0 knobs)
 *   • Issue #16 — Train Policy / Stop Training
 *   • Issue #17 — live stats overlay
 *   • Issue #20 — two behavior overlays during training:
 *       1. Coverage heatmap  — 10×10 grid shaded by agent visitation
 *       2. Velocity vectors  — per-agent direction arrows from frame diffs
 *
 * Returns a React fragment; App.tsx's .gym-stage flex column absorbs both
 * the controls bar and the .gym-scene without modification.
 */

import { useEffect, useRef, useState } from 'react'
import BattlefieldParamsPanel from './BattlefieldParamsPanel'
import { type BattlefieldParams, getScenarioDefaults } from './battlefieldParams'
import { type ScenarioCard, type ScenarioTelemetry, getScenarioById } from './scenarios'
import { SwarmEnv, GRID, WORLD_HALF } from '../swarm/sim'
import {
  checkPolicyExists,
  loadPolicy,
  type Policy,
  type PolicyStatus,
} from '../swarm/policy'
import { TrainingStatsDrawer, useTraining } from './TrainingDashboard'

// ──────────────────────────────────────────────── scene types ──────────────

interface Point { x: number; y: number }

interface Agent extends Point {
  id: string
  team?: 'blue' | 'red' | 'neutral'
  alive?: boolean
  radius?: number
  vx?: number
  vy?: number
}

interface Obstacle extends Point {
  id: string
  width?: number
  height?: number
  radius?: number
  rotation?: number
  kind?: 'barrier' | 'building' | 'crate' | 'jammer' | 'sensor' | 'vehicle'
}

interface Asset extends Point {
  id: string
  radius: number
}

interface Zone extends Point {
  id: string
  radius?: number
  width?: number
  height?: number
  kind: 'control' | 'jammer' | 'blue-territory' | 'red-territory' | 'exclusion' | 'search'
  avoid?: boolean
}

interface StageFrame {
  agents: Agent[]
  obstacles?: Obstacle[]
  zones?: Zone[]
  assets?: Asset[]
  paths?: Point[][]
  telemetry: ScenarioTelemetry[]
  ringRadius?: number
  contour?: Point[]
}

interface RolloutSnapshot {
  agents: Agent[]
  covered: Uint8Array
  coverage: number
  nAlive: number
  n: number
}

// ───────────────────────────────────────── scene animation helpers ─────────

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

function orbit(cx: number, cy: number, radius: number, angle: number, squash = 1) {
  return {
    x: cx + Math.cos(angle) * radius,
    y: cy + Math.sin(angle) * radius * squash,
  }
}

function repelFromRect(point: Point, rect: Obstacle | Zone, padding: number): Point {
  const width = rect.width ?? 0
  const height = rect.height ?? 0
  if (!width || !height) return point

  const halfW = width / 2 + padding
  const halfH = height / 2 + padding
  const dx = point.x - rect.x
  const dy = point.y - rect.y
  if (Math.abs(dx) >= halfW || Math.abs(dy) >= halfH) return point

  const pushX = halfW - Math.abs(dx)
  const pushY = halfH - Math.abs(dy)
  if (pushX < pushY) {
    return { x: rect.x + Math.sign(dx || 1) * halfW, y: point.y }
  }
  return { x: point.x, y: rect.y + Math.sign(dy || 1) * halfH }
}

function repelFromCircle(point: Point, circle: Obstacle | Zone | Asset, padding: number): Point {
  const radius = ('radius' in circle ? circle.radius : 0) ?? 0
  if (!radius) return point

  const dx = point.x - circle.x
  const dy = point.y - circle.y
  const dist = Math.max(Math.hypot(dx, dy), 0.001)
  const minDist = radius + padding
  if (dist >= minDist) return point

  return {
    x: circle.x + (dx / dist) * minDist,
    y: circle.y + (dy / dist) * minDist,
  }
}

function avoidScenarioObjects(agent: Agent, frame: StageFrame): Agent {
  if (agent.alive === false) return agent

  let next: Point = { x: agent.x, y: agent.y }
  const padding = (agent.radius ?? 3.2) + 2.2

  for (const obstacle of frame.obstacles ?? []) {
    next = obstacle.radius
      ? repelFromCircle(next, obstacle, padding)
      : repelFromRect(next, obstacle, padding)
  }

  for (const zone of frame.zones ?? []) {
    if (!zone.avoid) continue
    next = zone.radius
      ? repelFromCircle(next, zone, padding)
      : repelFromRect(next, zone, padding)
  }

  return {
    ...agent,
    x: clamp(next.x, 4, 96),
    y: clamp(next.y, 4, 96),
  }
}

function withAvoidance(frame: StageFrame): StageFrame {
  return {
    ...frame,
    agents: frame.agents.map((agent) => avoidScenarioObjects(agent, frame)),
  }
}

// ─────────────────────────────────────────── per-scenario renderers ────────

function renderDroneVsDrone(tick: number): StageFrame {
  const blueAngles = [0.2, 1.15, 2.2]
  const redAngles  = [3.5, 4.45, 5.2]
  const aliveRed   = tick > 88 ? 2 : 3
  const control    = clamp(50 + Math.sin(tick / 12) * 24, 8, 92)
  return {
    obstacles: [
      { id: 'blast-wall-a', x: 29, y: 50, width: 10, height: 32, kind: 'barrier' },
      { id: 'blast-wall-b', x: 71, y: 50, width: 10, height: 32, kind: 'barrier' },
      { id: 'radar-mast', x: 50, y: 24, radius: 5.5, kind: 'sensor' },
    ],
    zones: [
      { id: 'control-lane', x: 50, y: 50, radius: 16, kind: 'control' },
      { id: 'rf-denial-pocket', x: 50, y: 24, radius: 10, kind: 'jammer', avoid: true },
    ],
    assets: [{ id: 'control-zone', x: 50, y: 50, radius: 13 }],
    agents: [
      ...blueAngles.map((angle, i) => ({
        id: `blue-${i}`, team: 'blue' as const, alive: true,
        ...orbit(33, 50, 12 + i * 2.2, angle + tick * 0.045, 0.9),
      })),
      ...redAngles.map((angle, i) => ({
        id: `red-${i}`, team: 'red' as const, alive: i < aliveRed,
        ...orbit(67, 50, 12 + i * 2.4, angle - tick * 0.042, 0.85),
      })),
    ],
    telemetry: [
      { label: 'Blue alive', value: '3/3' },
      { label: 'Red alive',  value: `${aliveRed}/3` },
      { label: 'Control',    value: `${Math.round(control)}% blue` },
    ],
    ringRadius: 13,
  }
}

function renderMovingTargetTrack(tick: number): StageFrame {
  const targetA = { x: 20 + tick * 0.62,           y: 58 + Math.sin(tick / 10) * 12 }
  const targetB = { x: 76 - Math.cos(tick / 12) * 12, y: 34 + Math.sin(tick / 9) * 10 }
  const drones  = [
    orbit(targetA.x, targetA.y, 8,  tick * 0.08, 0.8),
    orbit(targetA.x, targetA.y, 14, tick * 0.05 + 1.4, 0.7),
    orbit(targetB.x, targetB.y, 8,  tick * 0.07 + 2.1, 0.8),
    orbit(targetB.x, targetB.y, 14, tick * 0.05 + 4.1, 0.7),
  ]
  const custody    = clamp(84 + Math.sin(tick / 8) * 12, 55, 99)
  const occlusions = 1 + (tick % 3 === 0 ? 1 : 0)
  return {
    obstacles: [
      { id: 'warehouse-a', x: 36, y: 30, width: 12, height: 24, rotation: -14, kind: 'building' },
      { id: 'warehouse-b', x: 57, y: 68, width: 14, height: 25, rotation: 12, kind: 'building' },
      { id: 'fuel-truck', x: 70, y: 44, width: 14, height: 7, rotation: -8, kind: 'vehicle' },
    ],
    zones: [
      { id: 'occlusion-shadow-a', x: 36, y: 30, radius: 14, kind: 'exclusion', avoid: true },
      { id: 'occlusion-shadow-b', x: 57, y: 68, radius: 15, kind: 'exclusion', avoid: true },
    ],
    paths: [
      Array.from({ length: 20 }, (_, s) => ({ x: 8 + s * 4.2, y: 58 + Math.sin((tick - s) / 10) * 12 })),
      Array.from({ length: 20 }, (_, s) => ({ x: 76 - Math.cos((tick - s) / 12) * 12, y: 34 + Math.sin((tick - s) / 9) * 10 })),
    ],
    agents: [
      { id: 'target-a', team: 'neutral', radius: 4.6, ...targetA },
      { id: 'target-b', team: 'neutral', radius: 4.6, ...targetB },
      ...drones.map((d, i) => ({ id: `tracker-${i}`, team: 'blue' as const, radius: 3.6, ...d })),
    ],
    telemetry: [
      { label: 'Targets tracked', value: '2 / 2' },
      { label: 'Occlusions',      value: `${occlusions}` },
      { label: 'Custody',         value: `${Math.round(custody)}%` },
    ],
  }
}

function renderSearchAndInterdict(tick: number): StageFrame {
  const lead   = orbit(52, 48, 22, tick * 0.03, 0.85)
  const wingA  = orbit(52, 48, 30, tick * 0.03 + 2.2, 0.75)
  const wingB  = orbit(52, 48, 30, tick * 0.03 + 4.1, 0.75)
  const closer = orbit(lead.x, lead.y, 10, tick * 0.06 + 0.6, 0.8)
  const lock   = clamp(38 + tick * 1.4, 0, 96)
  return {
    obstacles: [
      { id: 'crate-a', x: 25, y: 22, width: 12, height: 12, kind: 'crate' },
      { id: 'crate-b', x: 38, y: 63, width: 16, height: 10, rotation: -10, kind: 'crate' },
      { id: 'crate-c', x: 68, y: 34, width: 14, height: 14, rotation: 8, kind: 'crate' },
      { id: 'crate-d', x: 76, y: 72, width: 12, height: 18, kind: 'crate' },
      { id: 'jammer-node', x: 52, y: 47, radius: 7, kind: 'jammer' },
    ],
    zones: [
      { id: 'search-box', x: 50, y: 50, width: 76, height: 72, kind: 'search' },
      { id: 'jammer-field', x: 52, y: 47, radius: 15, kind: 'jammer', avoid: true },
    ],
    contour: [{ x: 12, y: 14 }, { x: 88, y: 14 }, { x: 88, y: 86 }, { x: 12, y: 86 }],
    agents: [
      { id: 'search-lead',  team: 'red',  radius: 4,   ...lead },
      { id: 'interdict-0',  team: 'blue',              ...wingA },
      { id: 'interdict-1',  team: 'blue',              ...wingB },
      { id: 'interdict-2',  team: 'blue',              ...closer },
    ],
    telemetry: [
      { label: 'Cells swept',   value: `${Math.round(clamp(22 + tick * 2.1, 22, 97))}%` },
      { label: 'Threat lock',   value: `${Math.round(lock)}%` },
      { label: 'Intercept ETA', value: `${Math.max(3, 18 - Math.floor(tick / 6))}s` },
    ],
  }
}

function renderDefendAsset(tick: number): StageFrame {
  const shieldAngles  = [0, 1.6, 3.1, 4.7]
  const inboundAngles = [0.6, 2.7, 5.05]
  const breaches      = tick > 94 ? 1 : 0
  const integrity     = clamp(100 - tick * 0.4 - breaches * 12, 63, 100)
  return {
    zones: [
      { id: 'standoff-ring', x: 50, y: 50, radius: 22, kind: 'exclusion' },
      { id: 'asset-inner-no-fly', x: 50, y: 50, radius: 11, kind: 'exclusion', avoid: true },
    ],
    assets: [{ id: 'asset', x: 50, y: 50, radius: 7 }],
    obstacles: [
      { id: 'hardpoint-north', x: 50, y: 26, width: 16, height: 6, kind: 'barrier' },
      { id: 'hardpoint-south', x: 50, y: 74, width: 16, height: 6, kind: 'barrier' },
      { id: 'generator', x: 64, y: 50, radius: 5, kind: 'sensor' },
    ],
    agents: [
      ...shieldAngles.map((angle, i) => ({
        id: `shield-${i}`, team: 'blue' as const,
        ...orbit(50, 50, 17, angle + tick * 0.035, 0.95),
      })),
      ...inboundAngles.map((angle, i) => ({
        id: `inbound-${i}`, team: 'red' as const, radius: 3.4,
        ...orbit(50, 50, 42 - tick * 0.18 + i * 2, angle, 1),
      })),
    ],
    telemetry: [
      { label: 'Breaches',         value: `${breaches}` },
      { label: 'Shield integrity', value: `${Math.round(integrity)}%` },
      { label: 'Interceptors',     value: '4 active' },
    ],
    ringRadius: 20,
  }
}

function renderCoverageRace(tick: number): StageFrame {
  const blueScore = clamp(24 + tick * 1.2, 24, 96)
  const redScore  = clamp(21 + tick * 1.08, 21, 91)
  const contested = clamp(18 + Math.sin(tick / 7) * 8, 4, 28)
  return {
    obstacles: [
      { id: 'jammer-a', x: 36, y: 46, width: 10, height: 18, rotation: 10, kind: 'jammer' },
      { id: 'jammer-b', x: 64, y: 52, width: 10, height: 18, rotation: -10, kind: 'jammer' },
      { id: 'score-gate-a', x: 50, y: 24, width: 28, height: 5, kind: 'barrier' },
      { id: 'score-gate-b', x: 50, y: 76, width: 28, height: 5, kind: 'barrier' },
    ],
    zones: [
      { id: 'blue-territory', x: 25, y: 50, width: 42, height: 78, kind: 'blue-territory' },
      { id: 'red-territory', x: 75, y: 50, width: 42, height: 78, kind: 'red-territory' },
      { id: 'jammer-field-a', x: 36, y: 46, radius: 12, kind: 'jammer', avoid: true },
      { id: 'jammer-field-b', x: 64, y: 52, radius: 12, kind: 'jammer', avoid: true },
    ],
    agents: [
      ...[0, 1, 2].map(i => ({ id: `race-blue-${i}`, team: 'blue' as const, ...orbit(30, 52, 12 + i * 5, tick * 0.032 + i, 0.8) })),
      ...[0, 1, 2].map(i => ({ id: `race-red-${i}`,  team: 'red'  as const, ...orbit(70, 48, 12 + i * 5, -tick * 0.032 + i, 0.8) })),
    ],
    telemetry: [
      { label: 'Blue score', value: `${Math.round(blueScore)}` },
      { label: 'Red score',  value: `${Math.round(redScore)}` },
      { label: 'Contested',  value: `${Math.round(contested)} cells` },
    ],
  }
}

function renderFrame(scenario: ScenarioCard, tick: number): StageFrame {
  const frame = (() => {
    switch (scenario.id) {
      case 'drone-vs-drone':      return renderDroneVsDrone(tick)
      case 'moving-target-track': return renderMovingTargetTrack(tick)
      case 'search-and-interdict':return renderSearchAndInterdict(tick)
      case 'defend-asset':        return renderDefendAsset(tick)
      case 'swarm-vs-swarm-race': return renderCoverageRace(tick)
      default:                    return renderSearchAndInterdict(tick)
    }
  })()
  return withAvoidance(frame)
}

function teamClass(team: Agent['team']) {
  if (team === 'red')     return 'gym-scene__agent gym-scene__agent--red'
  if (team === 'neutral') return 'gym-scene__agent gym-scene__agent--neutral'
  return 'gym-scene__agent gym-scene__agent--blue'
}

function renderZone(zone: Zone) {
  const className = `gym-scene__zone gym-scene__zone--${zone.kind}`
  if (zone.radius) {
    return (
      <circle
        key={zone.id}
        cx={zone.x}
        cy={zone.y}
        r={zone.radius}
        className={className}
      />
    )
  }
  return (
    <rect
      key={zone.id}
      x={zone.x - (zone.width ?? 0) / 2}
      y={zone.y - (zone.height ?? 0) / 2}
      width={zone.width ?? 0}
      height={zone.height ?? 0}
      rx="2"
      className={className}
    />
  )
}

function renderObstacle(obs: Obstacle) {
  const className = `gym-scene__obstacle gym-scene__obstacle--${obs.kind ?? 'barrier'}`
  if (obs.radius) {
    return (
      <circle
        key={obs.id}
        cx={obs.x}
        cy={obs.y}
        r={obs.radius}
        className={className}
      />
    )
  }
  return (
    <rect
      key={obs.id}
      x={obs.x - (obs.width ?? 0) / 2}
      y={obs.y - (obs.height ?? 0) / 2}
      width={obs.width ?? 0}
      height={obs.height ?? 0}
      rx="2.5"
      className={className}
      transform={obs.rotation ? `rotate(${obs.rotation} ${obs.x} ${obs.y})` : undefined}
    />
  )
}

// ──────────────────────────────────── overlay helpers (#20) ───────────────

const HEATMAP_ROWS = 10
const HEATMAP_COLS = 10

function emptyGrid(): number[][] {
  return Array.from({ length: HEATMAP_ROWS }, () => Array(HEATMAP_COLS).fill(0))
}

function worldToStage(value: number) {
  return clamp(((value + WORLD_HALF) / (WORLD_HALF * 2)) * 100, 4, 96)
}

function makeRolloutSnapshot(env: SwarmEnv): RolloutSnapshot {
  return {
    agents: Array.from({ length: env.n }, (_, i) => ({
      id: `policy-${i}`,
      team: 'blue' as const,
      alive: env.alive[i],
      x: worldToStage(env.pos[i * 2]),
      y: worldToStage(env.pos[i * 2 + 1]),
      radius: 3.4,
      vx: env.vel[i * 2] * 5,
      vy: env.vel[i * 2 + 1] * 5,
    })),
    covered: new Uint8Array(env.coveredCells()),
    coverage: env.coverage(),
    nAlive: env.nAlive(),
    n: env.n,
  }
}

function applyPolicyRollout(base: StageFrame, snapshot: RolloutSnapshot): StageFrame {
  return withAvoidance({
    ...base,
    agents: snapshot.agents,
    telemetry: [
      { label: 'Cells swept', value: `${Math.round(snapshot.coverage * 100)}%` },
      { label: 'Policy rollout', value: `${snapshot.nAlive} / ${snapshot.n} active` },
      { label: 'Controller', value: 'ONNX' },
    ],
  })
}

function coverageToHeatmap(snapshot: RolloutSnapshot): number[][] {
  const grid = emptyGrid()

  for (let cx = 0; cx < GRID; cx++) {
    for (let cy = 0; cy < GRID; cy++) {
      if (!snapshot.covered[cx * GRID + cy]) continue
      const col = Math.min(Math.floor((cx / GRID) * HEATMAP_COLS), HEATMAP_COLS - 1)
      const row = Math.min(Math.floor((cy / GRID) * HEATMAP_ROWS), HEATMAP_ROWS - 1)
      grid[row][col] = Math.min(grid[row][col] + 0.28, 1)
    }
  }

  return grid
}

function makeVisualizationParams(params: BattlefieldParams): BattlefieldParams {
  return {
    ...params,
    logistics: {
      ...params.logistics,
      attritionInjectRate: 0,
    },
  }
}

// ─────────────────────────────────────────────── component ────────────────

export interface GymScenarioStageProps {
  scenario: ScenarioCard
  policyStatus?: PolicyStatus
  onPolicyReady?: (envId: string) => void
  onTrainingStart?: (envId: string) => void
  onTrainingError?: (message: string) => void
}

export default function GymScenarioStage({
  scenario,
  policyStatus = 'not-trained',
  onPolicyReady,
  onTrainingStart,
  onTrainingError,
}: GymScenarioStageProps) {
  // ── animation tick ──────────────────────────────────────────────────────
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = window.setInterval(() => setTick(v => v + 1), 120)
    return () => window.clearInterval(id)
  }, [scenario.id])

  // ── battlefield params (#25) ────────────────────────────────────────────
  const [params, setParams]       = useState(() => getScenarioDefaults(scenario.id))
  const [paramsOpen, setParamsOpen] = useState(false)

  // ── training (#16) ──────────────────────────────────────────────────────
  const { status, metrics, start, stop } = useTraining(scenario.id, params, {
    onComplete: async envId => {
      const exists = await checkPolicyExists(envId)
      if (exists) onPolicyReady?.(envId)
    },
    onError: msg => onTrainingError?.(msg),
  })
  const isTraining = status === 'running'
  const policyReady = policyStatus === 'ready' || status === 'completed'

  const handleTrain = () => {
    onTrainingStart?.(scenario.id)
    void start()
  }

  // ── post-training policy rollout ────────────────────────────────────────
  const rolloutEnvRef = useRef<SwarmEnv | null>(null)
  const rolloutPolicyRef = useRef<Policy | null>(null)
  const rolloutSteppingRef = useRef(false)
  const [rolloutSnapshot, setRolloutSnapshot] = useState<RolloutSnapshot | null>(null)

  useEffect(() => {
    rolloutEnvRef.current = new SwarmEnv(400, 11, makeVisualizationParams(params))
    rolloutPolicyRef.current = null
    rolloutSteppingRef.current = false
    queueMicrotask(() => setRolloutSnapshot(null))
  }, [scenario.id, params])

  useEffect(() => {
    if (!policyReady) {
      rolloutPolicyRef.current = null
      return
    }

    let cancelled = false
    loadPolicy(scenario.id)
      .then(policy => {
        if (!cancelled) {
          rolloutPolicyRef.current = policy
          if (!rolloutEnvRef.current) {
            rolloutEnvRef.current = new SwarmEnv(400, 11, makeVisualizationParams(params))
          }
          rolloutEnvRef.current.reset(11)
          setRolloutSnapshot(makeRolloutSnapshot(rolloutEnvRef.current))
        }
      })
      .catch(err => {
        console.error('[gym] failed to load trained policy rollout:', err)
        rolloutPolicyRef.current = null
      })

    return () => {
      cancelled = true
    }
  }, [policyReady, scenario.id, params])

  useEffect(() => {
    const env = rolloutEnvRef.current
    const policy = rolloutPolicyRef.current
    if (!policyReady || !env || !policy || rolloutSteppingRef.current) return

    rolloutSteppingRef.current = true
    void policy
      .act(env.observe(), env.n)
      .then(actions => {
        env.step(actions)
        if (env.steps >= env.maxSteps) env.reset(11 + tick)
        setRolloutSnapshot(makeRolloutSnapshot(env))
      })
      .catch(err => {
        console.error('[gym] trained policy rollout failed:', err)
        rolloutPolicyRef.current = null
      })
      .finally(() => {
        rolloutSteppingRef.current = false
      })
  }, [tick, policyReady])

  // ── overlay #1: coverage heatmap (#20) ──────────────────────────────────
  const heatmapRef = useRef<number[][]>(emptyGrid())
  const [heatmap, setHeatmap] = useState<readonly (readonly number[])[]>(emptyGrid)
  const visibleHeatmap = rolloutSnapshot ? coverageToHeatmap(rolloutSnapshot) : heatmap

  // Reset heatmap when a new training run starts
  const prevStatusRef = useRef(status)
  useEffect(() => {
    if (status === 'running' && prevStatusRef.current !== 'running') {
      heatmapRef.current = emptyGrid()
    }
    prevStatusRef.current = status
  }, [status])

  // Accumulate agent positions → heatmap every tick while training/rollout
  useEffect(() => {
    if (status !== 'running') return
    const frame = renderFrame(scenario, tick % 120)
    const grid  = heatmapRef.current
    frame.agents.forEach(agent => {
      if (agent.alive === false) return
      const col = Math.min(Math.floor(agent.x / 10), HEATMAP_COLS - 1)
      const row = Math.min(Math.floor(agent.y / 10), HEATMAP_ROWS - 1)
      grid[row][col] = Math.min(grid[row][col] + 0.22, 1)
    })
    // Slow decay so trail gradually fades
    for (let r = 0; r < HEATMAP_ROWS; r++) {
      for (let c = 0; c < HEATMAP_COLS; c++) {
        if (grid[r][c] > 0) grid[r][c] = Math.max(0, grid[r][c] - 0.003)
      }
    }
    setHeatmap(grid.map(row => [...row]))
  }, [tick, status, scenario])

  let frame = rolloutSnapshot
    ? applyPolicyRollout(renderFrame(scenario, tick % 120), rolloutSnapshot)
    : renderFrame(scenario, tick % 120)
  const showBehaviorOverlays = isTraining || Boolean(rolloutSnapshot)

  if (!rolloutSnapshot && isTraining && params.weather.windSpeed > 0) {
    const drift = params.weather.windSpeed * 0.04
    const dx = Math.cos(params.weather.windDirRad) * drift
    const dy = Math.sin(params.weather.windDirRad) * drift
    frame = {
      ...frame,
      agents: frame.agents.map(a => ({
        ...a,
        x: clamp(a.x + dx, 4, 96),
        y: clamp(a.y + dy, 4, 96),
      })),
    }
    frame = withAvoidance(frame)
  }

  // Policy rollouts carry action vectors from SwarmEnv; scripted previews omit them.
  const velocities = frame.agents.map(agent => {
    return {
      id:  agent.id,
      x:   agent.x,
      y:   agent.y,
      vx:  agent.vx ?? 0,
      vy:  agent.vy ?? 0,
    }
  })

  // ──────────────────────────────────────────────────────── render ─────────

  return (
    <>
      {/* ── #25 params panel + #16 train button ─────────────────────── */}
      <BattlefieldParamsPanel
        params={params}
        onChange={setParams}
        isTraining={isTraining}
        onTrain={handleTrain}
        onStop={() => void stop()}
        open={paramsOpen}
        onToggleOpen={() => setParamsOpen(v => !v)}
      />

      {/* ── scene canvas ─────────────────────────────────────────────── */}
      <div className={`gym-scene${scenario.id === 'drone-vs-drone' ? ' gym-scene--flat' : ''}`} aria-label={`${scenario.name} simulated stage`}>
        <svg
          className="gym-scene__svg"
          viewBox="0 0 100 100"
          role="img"
          aria-label={scenario.summary}
        >
          <defs>
            {/* grid pattern */}
            <pattern id="gym-grid" width="8" height="8" patternUnits="userSpaceOnUse">
              <path d="M 8 0 L 0 0 0 8" fill="none" className="gym-scene__grid-line" />
            </pattern>

            {/* arrowhead marker for velocity vectors */}
            <marker
              id="gym-arrow"
              markerWidth="5"
              markerHeight="5"
              refX="4"
              refY="2.5"
              orient="auto"
            >
              <path d="M0,0 L0,5 L5,2.5 z" fill="rgba(255, 210, 90, 0.82)" />
            </marker>
          </defs>

          {/* floor */}
          <rect x="0" y="0" width="100" height="100" className="gym-scene__floor" />
          <rect x="0" y="0" width="100" height="100" fill="url(#gym-grid)" />

          {/* ── overlay #1: coverage heatmap ─────────────────────────── */}
          {showBehaviorOverlays && (
            <g aria-label="Coverage heatmap overlay">
              {(visibleHeatmap as number[][]).flatMap((row, r) =>
                row.map((intensity, c) =>
                  intensity > 0.04 ? (
                    <rect
                      key={`hm-${r}-${c}`}
                      x={c * 10}
                      y={r * 10}
                      width={10}
                      height={10}
                      fill={`rgba(113, 215, 255, ${(intensity * 0.55).toFixed(3)})`}
                    />
                  ) : null,
                ),
              )}
            </g>
          )}

          {/* scenario-specific geometry */}
          {frame.contour ? (
            <polygon
              points={frame.contour.map(p => `${p.x},${p.y}`).join(' ')}
              className="gym-scene__boundary"
            />
          ) : null}

          {frame.ringRadius ? (
            <circle cx="50" cy="50" r={frame.ringRadius} className="gym-scene__ring" />
          ) : null}

          {frame.zones?.map(renderZone)}

          {frame.assets?.map(asset => (
            <g key={asset.id}>
              <circle cx={asset.x} cy={asset.y} r={asset.radius}     className="gym-scene__asset" />
              <circle cx={asset.x} cy={asset.y} r={asset.radius + 5} className="gym-scene__asset-halo" />
            </g>
          ))}

          {frame.paths?.map((path, i) => (
            <polyline
              key={`path-${i}`}
              points={path.map(p => `${p.x},${p.y}`).join(' ')}
              className="gym-scene__path"
            />
          ))}

          {frame.obstacles?.map(renderObstacle)}

          {frame.agents.map(agent => (
            <g key={agent.id} opacity={agent.alive === false ? 0.28 : 1}>
              <circle
                cx={agent.x} cy={agent.y}
                r={(agent.radius ?? 3.2) + 2.5}
                className="gym-scene__agent-halo"
              />
              <circle
                cx={agent.x} cy={agent.y}
                r={agent.radius ?? 3.2}
                className={teamClass(agent.team)}
              />
            </g>
          ))}

          {/* ── overlay #2: velocity vectors ─────────────────────────── */}
          {showBehaviorOverlays && (
            <g aria-label="Velocity vector overlay">
              {velocities.map(v => {
                const speed = Math.sqrt(v.vx * v.vx + v.vy * v.vy)
                if (speed < 0.5) return null
                return (
                  <line
                    key={`vel-${v.id}`}
                    x1={v.x}
                    y1={v.y}
                    x2={v.x + v.vx}
                    y2={v.y + v.vy}
                    stroke="rgba(255, 210, 90, 0.78)"
                    strokeWidth="0.9"
                    markerEnd="url(#gym-arrow)"
                  />
                )
              })}
            </g>
          )}
        </svg>

        {/* ── #17 live stats overlay ───────────────────────────────────── */}
        <TrainingStatsDrawer metrics={metrics} status={status} />
      </div>
    </>
  )
}

export interface ScenarioMiniPreviewProps {
  scenarioId: string
  className?: string
}

export function ScenarioMiniPreview({ scenarioId, className }: ScenarioMiniPreviewProps) {
  const [tick, setTick] = useState(() => Math.floor(Math.random() * 120))

  useEffect(() => {
    const id = window.setInterval(() => setTick(v => (v + 1) % 120), 120)
    return () => window.clearInterval(id)
  }, [scenarioId])

  const scenario = getScenarioById(scenarioId)
  const frame = renderFrame(scenario, tick)

  return (
    <div className={`gym-mini-preview ${className ?? ''}`}>
      <svg
        className="gym-mini-preview__svg"
        viewBox="0 0 100 100"
        role="img"
        aria-label={scenario.summary}
      >
        <defs>
          <pattern id={`mini-grid-${scenarioId}`} width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" className="gym-scene__grid-line" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
          </pattern>
        </defs>

        <rect x="0" y="0" width="100" height="100" className="gym-scene__floor" fill="#080f17" />
        <rect x="0" y="0" width="100" height="100" fill={`url(#mini-grid-${scenarioId})`} />

        {/* boundary geometry */}
        {frame.contour ? (
          <polygon
            points={frame.contour.map(p => `${p.x},${p.y}`).join(' ')}
            className="gym-scene__boundary"
          />
        ) : null}

        {frame.ringRadius ? (
          <circle cx="50" cy="50" r={frame.ringRadius} className="gym-scene__ring" />
        ) : null}

        {frame.zones?.map(renderZone)}

        {frame.assets?.map(asset => (
          <g key={asset.id}>
            <circle cx={asset.x} cy={asset.y} r={asset.radius} className="gym-scene__asset" />
            <circle cx={asset.x} cy={asset.y} r={asset.radius + 5} className="gym-scene__asset-halo" />
          </g>
        ))}

        {frame.paths?.map((path, i) => (
          <polyline
            key={`path-${i}`}
            points={path.map(p => `${p.x},${p.y}`).join(' ')}
            className="gym-scene__path"
          />
        ))}

        {frame.obstacles?.map(renderObstacle)}

        {frame.agents.map(agent => (
          <g key={agent.id} opacity={agent.alive === false ? 0.28 : 1}>
            <circle
              cx={agent.x} cy={agent.y}
              r={(agent.radius ?? 3.2) + 2.5}
              className="gym-scene__agent-halo"
            />
            <circle
              cx={agent.x} cy={agent.y}
              r={agent.radius ?? 3.2}
              className={teamClass(agent.team)}
            />
          </g>
        ))}
      </svg>
    </div>
  )
}
