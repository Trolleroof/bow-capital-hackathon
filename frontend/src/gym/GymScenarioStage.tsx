/**
 * GymScenarioStage.tsx
 *
 * Owns the full gym stage area:
 *   • Scripted scene animation (unchanged)
 *   • Issue #25 — BattlefieldParamsPanel (P0 knobs, garrison/combat presets)
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
import { getScenarioDefaults } from './battlefieldParams'
import type { ScenarioCard, ScenarioTelemetry } from './scenarios'
import { TrainingStatsDrawer, useTraining } from './TrainingDashboard'

// ──────────────────────────────────────────────── scene types ──────────────

interface Point { x: number; y: number }

interface Agent extends Point {
  id: string
  team?: 'blue' | 'red' | 'neutral'
  alive?: boolean
  radius?: number
}

interface Obstacle extends Point {
  id: string
  width: number
  height: number
  rotation?: number
}

interface Asset extends Point {
  id: string
  radius: number
}

interface StageFrame {
  agents: Agent[]
  obstacles?: Obstacle[]
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

// ─────────────────────────────────────────── per-scenario renderers ────────

function renderDroneVsDrone(tick: number): StageFrame {
  const blueAngles = [0.2, 1.15, 2.2]
  const redAngles  = [3.5, 4.45, 5.2]
  const aliveRed   = tick > 88 ? 2 : 3
  const control    = clamp(50 + Math.sin(tick / 12) * 24, 8, 92)
  return {
    obstacles: [
      { id: 'wall-a', x: 29, y: 50, width: 10, height: 32 },
      { id: 'wall-b', x: 71, y: 50, width: 10, height: 32 },
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
      { id: 'stack-a', x: 44, y: 28, width: 10, height: 18, rotation: -14 },
      { id: 'stack-b', x: 55, y: 68, width: 12, height: 22, rotation: 12 },
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
      { id: 'crate-a', x: 25, y: 22, width: 12, height: 12 },
      { id: 'crate-b', x: 38, y: 63, width: 16, height: 10, rotation: -10 },
      { id: 'crate-c', x: 68, y: 34, width: 14, height: 14, rotation: 8 },
      { id: 'crate-d', x: 76, y: 72, width: 12, height: 18 },
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
    assets: [{ id: 'asset', x: 50, y: 50, radius: 7 }],
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
      { id: 'jammer-a', x: 36, y: 46, width: 10, height: 18, rotation: 10 },
      { id: 'jammer-b', x: 64, y: 52, width: 10, height: 18, rotation: -10 },
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
  switch (scenario.id) {
    case 'drone-vs-drone':      return renderDroneVsDrone(tick)
    case 'moving-target-track': return renderMovingTargetTrack(tick)
    case 'search-and-interdict':return renderSearchAndInterdict(tick)
    case 'defend-asset':        return renderDefendAsset(tick)
    case 'swarm-vs-swarm-race': return renderCoverageRace(tick)
    default:                    return renderSearchAndInterdict(tick)
  }
}

function teamClass(team: Agent['team']) {
  if (team === 'red')     return 'gym-scene__agent gym-scene__agent--red'
  if (team === 'neutral') return 'gym-scene__agent gym-scene__agent--neutral'
  return 'gym-scene__agent gym-scene__agent--blue'
}

// ──────────────────────────────────── overlay helpers (#20) ───────────────

const HEATMAP_ROWS = 10
const HEATMAP_COLS = 10

function emptyGrid(): number[][] {
  return Array.from({ length: HEATMAP_ROWS }, () => Array(HEATMAP_COLS).fill(0))
}

// ─────────────────────────────────────────────── component ────────────────

export default function GymScenarioStage({ scenario }: { scenario: ScenarioCard }) {
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
  const { status, metrics, start, stop } = useTraining(scenario.id, params)
  const isTraining = status === 'running'

  // ── overlay #1: coverage heatmap (#20) ──────────────────────────────────
  const heatmapRef = useRef<number[][]>(emptyGrid())
  const [heatmap, setHeatmap] = useState<readonly (readonly number[])[]>(emptyGrid)

  // Reset heatmap when a new training run starts
  const prevStatusRef = useRef(status)
  useEffect(() => {
    if (status === 'running' && prevStatusRef.current !== 'running') {
      heatmapRef.current = emptyGrid()
    }
    prevStatusRef.current = status
  }, [status])

  // Accumulate agent positions → heatmap every tick while training
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

  // ── overlay #2: velocity vectors (#20) ──────────────────────────────────
  const prevAgentsRef = useRef<Map<string, Point>>(new Map())

  const frame = renderFrame(scenario, tick % 120)

  // Compute per-agent velocity from frame diff (scaled for visibility)
  const velocities = frame.agents.map(agent => {
    const prev = prevAgentsRef.current.get(agent.id)
    return {
      id:  agent.id,
      x:   agent.x,
      y:   agent.y,
      vx:  prev ? (agent.x - prev.x) * 9 : 0,
      vy:  prev ? (agent.y - prev.y) * 9 : 0,
    }
  })

  // Update prev-agents snapshot after every render
  useEffect(() => {
    const map = new Map<string, Point>()
    frame.agents.forEach(a => map.set(a.id, { x: a.x, y: a.y }))
    prevAgentsRef.current = map
  })

  // ──────────────────────────────────────────────────────── render ─────────

  return (
    <>
      {/* ── #25 params panel + #16 train button ─────────────────────── */}
      <BattlefieldParamsPanel
        params={params}
        onChange={setParams}
        isTraining={isTraining}
        onTrain={start}
        onStop={stop}
        open={paramsOpen}
        onToggleOpen={() => setParamsOpen(v => !v)}
      />

      {/* ── scene canvas ─────────────────────────────────────────────── */}
      <div className="gym-scene" aria-label={`${scenario.name} simulated stage`}>
        {/* HUD telemetry chips (top-left) */}
        <div className="gym-scene__hud">
          {frame.telemetry.map(item => (
            <article key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </article>
          ))}
        </div>

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
          {isTraining && (
            <g aria-label="Coverage heatmap overlay">
              {(heatmap as number[][]).flatMap((row, r) =>
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

          {frame.obstacles?.map(obs => (
            <rect
              key={obs.id}
              x={obs.x - obs.width / 2}
              y={obs.y - obs.height / 2}
              width={obs.width}
              height={obs.height}
              rx="2.5"
              className="gym-scene__obstacle"
              transform={obs.rotation ? `rotate(${obs.rotation} ${obs.x} ${obs.y})` : undefined}
            />
          ))}

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
          {isTraining && (
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
