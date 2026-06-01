/**
 * TrainingDashboard.tsx — live metrics from swarm/train_service WebSocket (#16–#17, #22).
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { BattlefieldParams } from './battlefieldParams'
import {
  DEFAULT_TRAIN_TIMESTEPS,
  TRAIN_WS_URL,
  type TrainEvent,
  fetchTrainStatus,
  startTraining,
  stopTraining,
} from './trainApi'

export interface TrainingMetrics {
  step: number
  episode: number
  reward: number
  actor_loss: number
  critic_loss: number
  entropy: number
  coverage: number
  task_score: number
  primary_metric: string
  primary_value: number
  n_alive: number
  params_hash: string
  phase: string
  bc_step: number
  bc_total: number
  bc_mse: number
}

export type TrainingStatus = 'idle' | 'running' | 'stopped' | 'completed' | 'failed'

function eventToMetrics(event: TrainEvent, swarmSize: number): TrainingMetrics {
  const losses = event.losses ?? {}
  const step = Number.isFinite(event.step) ? event.step : 0
  const taskMetrics = event.task_metrics ?? {}
  return {
    step,
    episode: Math.floor(step / 200),
    reward: event.reward_mean ?? 0,
    actor_loss: losses.pg_loss ?? 0,
    critic_loss: losses.v_loss ?? 0,
    entropy: losses.entropy ?? 0,
    coverage: event.coverage ?? 0,
    task_score: event.task_score ?? event.coverage ?? 0,
    primary_metric: event.primary_metric ?? 'coverage',
    primary_value: event.primary_value ?? event.task_score ?? event.coverage ?? 0,
    n_alive: swarmSize,
    params_hash: (event.params_hash ?? '').slice(0, 8).toUpperCase(),
    phase: event.phase ?? '',
    bc_step: taskMetrics.bc_step ?? 0,
    bc_total: taskMetrics.bc_total ?? 0,
    bc_mse: taskMetrics.bc_mse ?? 0,
  }
}

export interface UseTrainingOptions {
  onComplete?: (envId: string) => void
  onError?: (message: string) => void
}

export interface UseTrainingResult {
  status: TrainingStatus
  metrics: TrainingMetrics | null
  history: TrainingMetrics[]
  start: () => void
  stop: () => void
}

export function useTraining(
  envId: string,
  params: BattlefieldParams,
  options?: UseTrainingOptions,
): UseTrainingResult {
  const [status, setStatus] = useState<TrainingStatus>('idle')
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null)
  const [history, setHistory] = useState<TrainingMetrics[]>([])

  const wsRef = useRef<WebSocket | null>(null)
  const startingRef = useRef(false)
  const optionsRef = useRef(options)
  optionsRef.current = options

  const closeSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  const handleEvent = useCallback(
    (event: TrainEvent) => {
      if (event.env_id && event.env_id !== envId) {
        return
      }

      const m = eventToMetrics(event, params.logistics.swarmSize)
      setMetrics(m)
      setHistory(prev => {
        if (prev.length > 0 && prev[prev.length - 1]?.step === m.step) {
          return prev
        }
        const next = [...prev, m]
        return next.length > 80 ? next.slice(-80) : next
      })

      if (event.phase === 'export_failed') {
        setStatus('failed')
        closeSocket()
        Promise.resolve().then(() => {
          optionsRef.current?.onError?.(event.error ?? 'ONNX export failed')
        })
        return
      }

      if (event.phase === 'final' || event.phase === 'exported') {
        setStatus('completed')
        closeSocket()
        Promise.resolve().then(() => {
          optionsRef.current?.onComplete?.(envId)
        })
      }
    },
    [envId, params.logistics.swarmSize, closeSocket],
  )

  const attachSocket = useCallback((): Promise<WebSocket> => {
    closeSocket()
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(TRAIN_WS_URL)
      wsRef.current = ws

      ws.onopen = () => resolve(ws)

      ws.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(String(ev.data)) as TrainEvent
          if (parsed.topic === 'train') handleEvent(parsed)
        } catch {
          /* ignore malformed frames */
        }
      }

      ws.onerror = () => {
        reject(new Error('WebSocket connection to training service failed'))
      }

      ws.onclose = () => {
        wsRef.current = null
      }
    })
  }, [closeSocket, handleEvent])

  const start = useCallback(async () => {
    if (startingRef.current) return
    startingRef.current = true
    try {
      setMetrics(null)
      setHistory([])
      setStatus('running')
      closeSocket()

      // Connect metrics socket in parallel — never block training on WS.
      // train_service replays the latest event when a client connects late.
      void attachSocket().catch(() => {
        console.warn(
          `[training] WebSocket unavailable (${TRAIN_WS_URL}); falling back to status polling.`,
        )
      })

      const { ok, error } = await startTraining(envId, 'combat')
      if (!ok) {
        closeSocket()
        setStatus('failed')
        Promise.resolve().then(() => {
          optionsRef.current?.onError?.(
            error ??
              'Training could not start. If a previous run is stuck, restart train_service or wait for it to finish.',
          )
        })
        return
      }
    } catch (err) {
      closeSocket()
      setStatus('failed')
      Promise.resolve().then(() => {
        optionsRef.current?.onError?.(
          err instanceof Error
            ? err.message
            : 'Training service request failed. Confirm swarm.backend is running on 127.0.0.1:8787.',
        )
      })
    } finally {
      startingRef.current = false
    }
  }, [envId, closeSocket, attachSocket])

  const stop = useCallback(async () => {
    closeSocket()
    await stopTraining(envId)
    setStatus('stopped')
  }, [envId, closeSocket])

  useEffect(() => {
    let canceled = false

    void fetchTrainStatus(envId).then(({ running, last }) => {
      if (canceled || !running) return
      setStatus('running')
      if (last?.topic === 'train') {
        handleEvent(last)
      }
      void attachSocket().catch(() => {
        /* status polling covers metrics if WebSocket fails */
      })
    })

    return () => {
      canceled = true
    }
  }, [envId, attachSocket, handleEvent])

  // Poll status while training so metrics still update if WebSocket drops.
  // Also poll in non-running states to detect new runs started externally.
  useEffect(() => {
    const interval = status === 'running' ? 2000 : 3000

    const poll = () => {
      void fetchTrainStatus(envId).then(({ running, last, status: jobStatus }) => {
        if (running && status !== 'running') {
          // New run detected while dashboard was idle/completed/stopped.
          setStatus('running')
          if (last?.topic === 'train') handleEvent(last)
          void attachSocket().catch(() => { /* polling fallback */ })
          return
        }

        if (last?.topic === 'train') {
          handleEvent(last)
        }
        if (status === 'running' && !running) {
          closeSocket()
          if (
            last?.phase === 'final' ||
            last?.phase === 'exported' ||
            last?.phase === 'export_failed'
          ) {
            return
          }
          if (jobStatus === 'completed') {
            setStatus('completed')
            optionsRef.current?.onComplete?.(envId)
          } else if (jobStatus === 'failed') {
            setStatus('failed')
          } else {
            setStatus('stopped')
          }
        }
      })
    }

    poll()
    const id = window.setInterval(poll, interval)
    return () => window.clearInterval(id)
  }, [status, envId, handleEvent, closeSocket, attachSocket])

  useEffect(() => () => closeSocket(), [closeSocket])

  return { status, metrics, history, start, stop }
}

interface BehavioralCloningLoaderProps {
  metrics: TrainingMetrics | null
  status: TrainingStatus
}

/** Left-side loader shown while the policy is warm-started via behavioral cloning. */
export function BehavioralCloningLoader({ metrics, status }: BehavioralCloningLoaderProps) {
  if (status !== 'running' || !metrics || metrics.phase !== 'warm_start') return null

  const total = metrics.bc_total > 0 ? metrics.bc_total : 1
  const pct = Math.min(100, Math.max(0, Math.round((metrics.bc_step / total) * 100)))

  return (
    <div className="gym-bc-loader" aria-live="polite" aria-label="Behavioral cloning progress">
      <div className="gym-bc-loader__header">
        <span className="gym-bc-loader__spinner" aria-hidden="true" />
        <span className="gym-bc-loader__title">Behavioral Cloning</span>
        <span className="gym-bc-loader__pct">{pct}%</span>
      </div>
      <div className="gym-bc-loader__track">
        <div className="gym-bc-loader__fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="gym-bc-loader__meta">
        <span>step {metrics.bc_step.toLocaleString()} / {metrics.bc_total.toLocaleString()}</span>
        <span>mse {metrics.bc_mse.toFixed(4)}</span>
      </div>
    </div>
  )
}

interface TrainingStatsDrawerProps {
  metrics: TrainingMetrics | null
  status: TrainingStatus
}

export function TrainingStatsDrawer({ metrics, status }: TrainingStatsDrawerProps) {
  if (status === 'idle' || !metrics) return null

  const statusLabel =
    status === 'running'
      ? 'TRAINING'
      : status === 'completed'
        ? 'COMPLETE'
        : status === 'failed'
          ? 'FAILED'
          : 'STOPPED'

  const simFrames = Math.max(0, Math.round(metrics.step))
  const progress = DEFAULT_TRAIN_TIMESTEPS > 0
    ? Math.min(1, simFrames / DEFAULT_TRAIN_TIMESTEPS)
    : 0
  const objectiveScore = Math.round(Math.max(metrics.task_score, metrics.primary_value, 0) * 100)
  const mappedAo = Math.round(Math.max(0, Math.min(1, metrics.coverage)) * 100)
  const swarmOnline = Math.max(0, metrics.n_alive)
  const readiness = status === 'completed'
    ? 'Exported'
    : progress >= 0.75
      ? 'Final pass'
      : progress >= 0.35
        ? 'Learning'
        : 'Warming up'

  const chips = [
    { label: 'Sim frames', value: simFrames.toLocaleString(), accent: simFrames > 0 },
    { label: 'Mission score', value: `${objectiveScore}%`, accent: objectiveScore > 0 },
    { label: 'Engagements', value: metrics.primary_value.toFixed(2), accent: metrics.primary_value > 0 },
    { label: 'Mapped AO', value: `${mappedAo}%`, accent: mappedAo > 0 },
    { label: 'Swarm online', value: `${swarmOnline}/${swarmOnline}`, accent: swarmOnline > 0 },
    { label: 'Deploy gate', value: readiness, accent: status === 'completed' || progress >= 0.75 },
  ]

  return (
    <div className="gym-stats-overlay" aria-live="polite" aria-label="Training metrics">
      <div
        className={`gym-stat-chip ${status === 'running' ? 'gym-stat-chip--pulse' : ''} ${status === 'completed' ? 'gym-stat-chip--accent' : ''}`}
      >
        <span className="gym-stat-chip__label">Status</span>
        <strong className="gym-stat-chip__value">
          {status === 'running' && <span className="gym-training-dot" aria-hidden="true" />}
          {statusLabel}
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
