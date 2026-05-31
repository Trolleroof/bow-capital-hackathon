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

import { useEffect, useState } from 'react'
import BattlefieldParamsPanel from './BattlefieldParamsPanel'
import { getScenarioDefaults } from './battlefieldParams'
import { type ScenarioCard, type ScenarioTelemetry, getScenarioById } from './scenarios'
import {
  checkPolicyExists,
  type PolicyStatus,
} from '../swarm/policy'
import { TrainingStatsDrawer, useTraining } from './TrainingDashboard'
import { TrainingMetricsChart } from './TrainingMetricsChart'

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

function waypointLoop(points: Point[], tick: number, speed = 1): Point {
  if (points.length === 0) return { x: 50, y: 50 }
  if (points.length === 1) return points[0]

  const segment = (tick * speed) / 18
  const rawIndex = Math.floor(segment)
  const i = ((rawIndex % points.length) + points.length) % points.length
  const a = points[i]
  const b = points[(i + 1) % points.length]
  const t = segment - rawIndex
  const eased = t * t * (3 - 2 * t)

  return {
    x: a.x + (b.x - a.x) * eased,
    y: a.y + (b.y - a.y) * eased,
  }
}

function pathTrail(points: Point[], tick: number, length = 18, speed = 1): Point[] {
  return Array.from({ length }, (_, i) => waypointLoop(points, tick - (length - i) * 1.8, speed))
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
  const blueRoutes = [
    [{ x: 16, y: 36 }, { x: 34, y: 31 }, { x: 43, y: 44 }, { x: 39, y: 63 }, { x: 20, y: 66 }],
    [{ x: 18, y: 50 }, { x: 35, y: 39 }, { x: 45, y: 50 }, { x: 35, y: 61 }],
    [{ x: 16, y: 64 }, { x: 31, y: 70 }, { x: 43, y: 58 }, { x: 34, y: 40 }],
  ]
  const redRoutes = [
    [{ x: 84, y: 36 }, { x: 66, y: 31 }, { x: 57, y: 44 }, { x: 61, y: 63 }, { x: 80, y: 66 }],
    [{ x: 82, y: 50 }, { x: 65, y: 39 }, { x: 55, y: 50 }, { x: 65, y: 61 }],
    [{ x: 84, y: 64 }, { x: 69, y: 70 }, { x: 57, y: 58 }, { x: 66, y: 40 }],
  ]
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
    paths: [
      pathTrail(blueRoutes[0], tick, 16, 1.05),
      pathTrail(redRoutes[0], tick + 9, 16, 1.05),
    ],
    agents: [
      ...blueRoutes.map((route, i) => ({
        id: `blue-${i}`, team: 'blue' as const, alive: true,
        ...waypointLoop(route, tick + i * 10, 1.05),
      })),
      ...redRoutes.map((route, i) => ({
        id: `red-${i}`, team: 'red' as const, alive: i < aliveRed,
        ...waypointLoop(route, tick + i * 10 + 9, 1.05),
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
  const sweepA = [{ x: 16, y: 20 }, { x: 42, y: 20 }, { x: 60, y: 24 }, { x: 84, y: 24 }, { x: 84, y: 44 }, { x: 64, y: 46 }, { x: 34, y: 44 }, { x: 16, y: 46 }]
  const sweepB = [{ x: 18, y: 72 }, { x: 42, y: 76 }, { x: 60, y: 69 }, { x: 84, y: 66 }, { x: 82, y: 84 }, { x: 55, y: 84 }, { x: 28, y: 82 }]
  const pincer = [{ x: 18, y: 54 }, { x: 36, y: 52 }, { x: 47, y: 59 }, { x: 58, y: 58 }, { x: 68, y: 51 }, { x: 82, y: 51 }]
  const threat = [{ x: 79, y: 79 }, { x: 73, y: 65 }, { x: 64, y: 58 }, { x: 58, y: 53 }, { x: 53, y: 50 }]
  const lead   = waypointLoop(threat, tick, 0.86)
  const wingA  = waypointLoop(sweepA, tick, 1.08)
  const wingB  = waypointLoop(sweepB, tick + 10, 1)
  const closer = waypointLoop(pincer, tick + 18, 1.14)
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
    paths: [
      pathTrail(sweepA, tick, 22, 1.08),
      pathTrail(sweepB, tick + 10, 20, 1),
      pathTrail(pincer, tick + 18, 16, 1.14),
      pathTrail(threat, tick, 14, 0.86),
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

// ─────────────────────────────────────────────── component ────────────────

export interface GymScenarioStageProps {
  scenario: ScenarioCard
  policyStatus?: PolicyStatus
  canLaunchSim?: boolean
  onLaunchSim?: () => void
  onPolicyReady?: (envId: string) => void
  onTrainingStart?: (envId: string) => void
  onTrainingError?: (message: string) => void
}

export default function GymScenarioStage({
  scenario,
  policyStatus = 'not-trained',
  canLaunchSim = false,
  onLaunchSim,
  onPolicyReady,
  onTrainingStart,
  onTrainingError,
}: GymScenarioStageProps) {
  // ── battlefield params (#25) ────────────────────────────────────────────
  const [params, setParams]       = useState(() => getScenarioDefaults(scenario.id))
  const [paramsOpen, setParamsOpen] = useState(false)

  // ── training (#16) ──────────────────────────────────────────────────────
  const { status, metrics, history, start, stop } = useTraining(scenario.id, params, {
    onComplete: async envId => {
      const exists = await checkPolicyExists(envId)
      if (exists) onPolicyReady?.(envId)
    },
    onError: msg => onTrainingError?.(msg),
  })
  const isTraining = status === 'running'
  void policyStatus

  const handleTrain = () => {
    onTrainingStart?.(scenario.id)
    void start()
  }

  // ──────────────────────────────────────────────────────── render ─────────

  return (
    <>
      {/* ── #25 params panel + #16 train button ─────────────────────── */}
      <BattlefieldParamsPanel
        params={params}
        onChange={setParams}
        isTraining={isTraining}
        canLaunchSim={canLaunchSim}
        onLaunchSim={() => onLaunchSim?.()}
        onTrain={handleTrain}
        onStop={() => void stop()}
        open={paramsOpen}
        onToggleOpen={() => setParamsOpen(v => !v)}
      />

      <section className="gym-train-console" aria-label={`${scenario.name} training controls`}>
        <div className="gym-train-console__main">
          <span className="gym-train-console__eyebrow">{scenario.label}</span>
          <h2>{scenario.name}</h2>
          <p>{scenario.summary}</p>

          <div className="gym-train-console__grid" aria-label="Scenario contract">
            <article>
              <span>Observation</span>
              <strong>{scenario.observation}</strong>
            </article>
            <article>
              <span>Action</span>
              <strong>{scenario.action}</strong>
            </article>
            <article>
              <span>Reward</span>
              <strong>{scenario.reward}</strong>
            </article>
          </div>
        </div>

        <aside className="gym-train-console__side">
          <div className="gym-policy-card">
            <span>Policy</span>
            <strong>{policyStatus}</strong>
          </div>
          <div className="gym-policy-card">
            <span>PyBullet Environment</span>
            <strong>{canLaunchSim ? 'ready to launch' : 'locked until policy export'}</strong>
          </div>
          <TrainingStatsDrawer metrics={metrics} status={status} />
          <TrainingMetricsChart history={history} />
        </aside>
      </section>
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
