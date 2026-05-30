/**
 * TrainingDashboard.tsx
 *
 * Issue #16 — Train Policy / Stop Training button logic (mock stream).
 * Issue #17 — Live stats drawer: step, episode, reward, losses, coverage, n_alive, params_hash.
 *
 * When Track A ships the real SSE bus (#22), replace the setInterval block in
 * useTraining with an EventSource subscription on /api/train/stream?job_id=…
 * The TrainingMetrics shape already matches the #22 schema.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { BattlefieldParams } from './battlefieldParams'

// ─────────────────────────────────────── types (match #22 schema) ──────────

export interface TrainingMetrics {
  step: number
  episode: number
  reward: number
  actor_loss: number
  critic_loss: number
  entropy: number
  /** Fraction 0–1 of map cells visited by the swarm */
  coverage: number
  n_alive: number
  params_hash: string
}

export type TrainingStatus = 'idle' | 'running' | 'stopped'

// ──────────────────────────────────────────────── helpers ──────────────────

function djb2Hash(s: string): string {
  let h = 5381
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) + h + s.charCodeAt(i)) >>> 0
  }
  return h.toString(16).padStart(8, '0').toUpperCase()
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function noise(scale: number) {
  return (Math.random() - 0.5) * scale
}

const SCENARIO_TARGETS: Record<string, { reward: number; coverage: number }> = {
  'drone-vs-drone':       { reward: 1.4,  coverage: 0.68 },
  'moving-target-track':  { reward: 0.9,  coverage: 0.82 },
  'search-and-interdict': { reward: 1.1,  coverage: 0.92 },
  'defend-asset':         { reward: 0.8,  coverage: 0.71 },
  'swarm-vs-swarm-race':  { reward: 1.2,  coverage: 0.85 },
}

// ──────────────────────────────────────────── useTraining hook ─────────────

export interface UseTrainingResult {
  status: TrainingStatus
  metrics: TrainingMetrics | null
  history: TrainingMetrics[]
  start: () => void
  stop: () => void
}

/**
 * Mock training loop — replace setInterval body with real SSE from #22 when ready.
 *
 * To wire real backend:
 *   1. POST /api/train  { env_id, params }  → { job_id }
 *   2. Open EventSource /api/train/stream?job_id=…
 *   3. Parse each `data:` line as TrainingMetrics JSON
 *   4. Call setMetrics / setHistory exactly as done here
 */
export function useTraining(
  envId: string,
  params: BattlefieldParams,
): UseTrainingResult {
  const [status, setStatus] = useState<TrainingStatus>('idle')
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null)
  const [history, setHistory] = useState<TrainingMetrics[]>([])

  const stepRef    = useRef(0)
  const episodeRef = useRef(0)
  const timerRef   = useRef<ReturnType<typeof setInterval> | null>(null)

  const start = useCallback(() => {
    stepRef.current    = 0
    episodeRef.current = 0
    setMetrics(null)
    setHistory([])
    setStatus('running')
  }, [])

  const stop = useCallback(() => setStatus('stopped'), [])

  useEffect(() => {
    if (status !== 'running') {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      return
    }

    const target      = SCENARIO_TARGETS[envId] ?? { reward: 1.0, coverage: 0.75 }
    const paramsHash  = djb2Hash(JSON.stringify(params))

    // ── mock metric stream (250 ms cadence) ──────────────────────────────
    timerRef.current = setInterval(() => {
      stepRef.current += Math.floor(Math.random() * 8) + 1
      const step = stepRef.current

      // rough episode boundary every ~200 steps
      episodeRef.current = Math.floor(step / 200)

      const t = Math.min(step / 3000, 1) // convergence horizon

      const reward      = lerp(-2,   target.reward,    Math.pow(t, 0.7)) + noise(0.25)
      const actor_loss  = lerp(1.5,  0.12,             Math.pow(t, 0.6)) + noise(0.06)
      const critic_loss = lerp(2.0,  0.18,             Math.pow(t, 0.55)) + noise(0.1)
      const entropy     = lerp(0.95, 0.18,             Math.pow(t, 0.5))  + noise(0.04)
      const coverage    = Math.min(
        lerp(0, target.coverage, Math.pow(t, 0.8)) + noise(0.015),
        1,
      )
      const n_alive = Math.max(
        1,
        Math.round(
          params.logistics.swarmSize -
            params.logistics.attritionInjectRate *
            params.logistics.swarmSize *
            3 *
            Math.random(),
        ),
      )

      const m: TrainingMetrics = {
        step,
        episode:     episodeRef.current,
        reward:      Math.round(reward * 100) / 100,
        actor_loss:  Math.max(0, Math.round(actor_loss  * 1000) / 1000),
        critic_loss: Math.max(0, Math.round(critic_loss * 1000) / 1000),
        entropy:     Math.max(0, Math.round(entropy     * 1000) / 1000),
        coverage:    Math.max(0, Math.round(coverage    * 100)  / 100),
        n_alive,
        params_hash: paramsHash,
      }

      setMetrics(m)
      setHistory(prev => {
        const next = [...prev, m]
        return next.length > 80 ? next.slice(-80) : next
      })
    }, 250)

    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [status, envId, params])

  return { status, metrics, history, start, stop }
}

// ──────────────────────────────────── TrainingStatsDrawer component ────────

interface TrainingStatsDrawerProps {
  metrics: TrainingMetrics | null
  status: TrainingStatus
}

/**
 * Issue #17 — live stats overlay anchored to the bottom of .gym-scene.
 * Appears whenever status !== 'idle'.
 */
export function TrainingStatsDrawer({ metrics, status }: TrainingStatsDrawerProps) {
  if (status === 'idle' || !metrics) return null

  const chips = [
    { label: 'Step',        value: metrics.step.toLocaleString() },
    { label: 'Episode',     value: String(metrics.episode) },
    { label: 'Reward',      value: metrics.reward.toFixed(2),     accent: metrics.reward > 0 },
    { label: 'Actor ℒ',    value: metrics.actor_loss.toFixed(3) },
    { label: 'Critic ℒ',   value: metrics.critic_loss.toFixed(3) },
    { label: 'Coverage',    value: `${Math.round(metrics.coverage * 100)}%`, accent: true },
    { label: 'N Alive',     value: String(metrics.n_alive) },
    { label: 'Hash',        value: metrics.params_hash.slice(0, 7) },
  ]

  return (
    <div className="gym-stats-overlay" aria-live="polite" aria-label="Training metrics">
      {/* status chip */}
      <div className={`gym-stat-chip ${status === 'running' ? 'gym-stat-chip--pulse' : ''}`}>
        <span className="gym-stat-chip__label">Status</span>
        <strong className="gym-stat-chip__value">
          {status === 'running' && (
            <span className="gym-training-dot" aria-hidden="true" />
          )}
          {status === 'running' ? 'TRAINING' : 'STOPPED'}
        </strong>
      </div>

      {chips.map(chip => (
        <div key={chip.label} className="gym-stat-chip">
          <span className="gym-stat-chip__label">{chip.label}</span>
          <strong
            className={`gym-stat-chip__value ${chip.accent ? 'gym-stat-chip__value--accent' : ''}`}
          >
            {chip.value}
          </strong>
        </div>
      ))}
    </div>
  )
}
