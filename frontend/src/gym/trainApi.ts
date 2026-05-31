/** Training bridge — proxies to swarm/train_service.py (HTTP :8787, WS :8766). */

export const TRAIN_API_BASE =
  import.meta.env.VITE_TRAIN_API_URL ?? ''

export const TRAIN_WS_URL =
  import.meta.env.VITE_TRAIN_WS_URL ?? 'ws://127.0.0.1:8766'

export const PYBULLET_WS_URL =
  import.meta.env.VITE_PYBULLET_WS_URL ?? 'ws://127.0.0.1:8765'

export const DEFAULT_TRAIN_TIMESTEPS = Number(
  import.meta.env.VITE_TRAIN_TIMESTEPS ?? 100_000,
)

export interface TrainEvent {
  topic: 'train'
  env_id: string
  profile?: string
  phase: string
  step: number
  reward_mean: number
  coverage: number
  losses?: {
    pg_loss?: number
    v_loss?: number
    entropy?: number
    approx_kl?: number
  }
  params_hash?: string
  error?: string
}

export async function startTraining(
  envId: string,
  profile: 'garrison' | 'combat',
  timesteps = DEFAULT_TRAIN_TIMESTEPS,
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${TRAIN_API_BASE}/api/train/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ env_id: envId, profile, timesteps }),
  })
  const data = (await res.json()) as { ok?: boolean; error?: string }
  return { ok: Boolean(data.ok), error: data.error }
}

export async function stopTraining(envId: string): Promise<void> {
  await fetch(`${TRAIN_API_BASE}/api/train/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ env_id: envId }),
  })
}

export async function startPyBulletSim(
  envId: string,
  cameraMode: 'observer' | 'chase' | 'fpv' = 'observer',
  selectedDrone = 0,
): Promise<{ ok: boolean; wsUrl: string; error?: string }> {
  const res = await fetch(`${TRAIN_API_BASE}/api/sim/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      env_id: envId,
      policy: 'trained',
      camera_mode: cameraMode,
      selected_drone: selectedDrone,
    }),
  })
  const data = (await res.json()) as {
    ok?: boolean
    ws_url?: string
    error?: string
  }
  return {
    ok: Boolean(data.ok),
    wsUrl: data.ws_url ?? PYBULLET_WS_URL,
    error: data.error,
  }
}
