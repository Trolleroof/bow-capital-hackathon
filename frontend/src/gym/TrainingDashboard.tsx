/**
 * TrainingDashboard.tsx — live metrics from swarm/train_service WebSocket (#16–#17, #22).
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { BattlefieldParams } from './battlefieldParams'
import {
  TRAIN_WS_URL,
  type TrainEvent,
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
  n_alive: number
  params_hash: string
}

export type TrainingStatus = 'idle' | 'running' | 'stopped' | 'completed' | 'failed'

function eventToMetrics(event: TrainEvent, swarmSize: number): TrainingMetrics {
  const losses = event.losses ?? {}
  return {
    step: event.step,
    episode: Math.floor(event.step / 200),
    reward: event.reward_mean ?? 0,
    actor_loss: losses.pg_loss ?? 0,
    critic_loss: losses.v_loss ?? 0,
    entropy: losses.entropy ?? 0,
    coverage: event.coverage ?? 0,
    n_alive: swarmSize,
    params_hash: (event.params_hash ?? '').slice(0, 8).toUpperCase(),
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
      if (event.env_id && event.env_id !== envId) return

      const m = eventToMetrics(event, params.logistics.swarmSize)
      setMetrics(m)
      setHistory(prev => {
        const next = [...prev, m]
        return next.length > 80 ? next.slice(-80) : next
      })

      if (event.phase === 'export_failed') {
        setStatus('failed')
        optionsRef.current?.onError?.(event.error ?? 'ONNX export failed')
        closeSocket()
        return
      }

      if (event.phase === 'final' || event.phase === 'exported') {
        setStatus('completed')
        closeSocket()
        optionsRef.current?.onComplete?.(envId)
      }
    },
    [envId, params.logistics.swarmSize, closeSocket],
  )

  const start = useCallback(async () => {
    setMetrics(null)
    setHistory([])
    setStatus('running')
    closeSocket()

    const { ok, error } = await startTraining(envId, 'combat')
    if (!ok) {
      setStatus('failed')
      optionsRef.current?.onError?.(
        error ??
          'Training service not reachable. Restart with `cd frontend && bun dev` (needs `uv` installed), or run train_service manually.',
      )
      return
    }

    const ws = new WebSocket(TRAIN_WS_URL)
    wsRef.current = ws

    ws.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(String(ev.data)) as TrainEvent
        if (parsed.topic === 'train') handleEvent(parsed)
      } catch {
        /* ignore malformed frames */
      }
    }

    ws.onerror = () => {
      setStatus(prev => {
        if (prev === 'running') {
          optionsRef.current?.onError?.(
            'WebSocket connection to training service failed',
          )
          return 'failed'
        }
        return prev
      })
    }

    ws.onclose = () => {
      wsRef.current = null
    }
  }, [envId, params, closeSocket, handleEvent])

  const stop = useCallback(async () => {
    closeSocket()
    await stopTraining(envId)
    setStatus('stopped')
  }, [envId, closeSocket])

  useEffect(() => () => closeSocket(), [closeSocket])

  return { status, metrics, history, start, stop }
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

  const chips = [
    { label: 'Step', value: metrics.step.toLocaleString() },
    { label: 'Episode', value: String(metrics.episode) },
    { label: 'Reward', value: metrics.reward.toFixed(2), accent: metrics.reward > 0 },
    { label: 'Actor ℒ', value: metrics.actor_loss.toFixed(3) },
    { label: 'Critic ℒ', value: metrics.critic_loss.toFixed(3) },
    { label: 'Coverage', value: `${Math.round(metrics.coverage * 100)}%`, accent: true },
    { label: 'N Alive', value: String(metrics.n_alive) },
    { label: 'Hash', value: metrics.params_hash.slice(0, 7) },
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
