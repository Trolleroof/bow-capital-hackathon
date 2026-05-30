/**
 * Starts swarm/train_service.py when Vite dev server boots so Train Policy works
 * with only `bun dev` (no separate terminal).
 *
 * If :8787 is already alive at startup it's reused, but a background watchdog
 * respawns the service if it later dies (covers the crash-after-probe case).
 * Disable auto-start: VITE_AUTO_TRAIN_SERVICE=0 bun dev
 */

import { spawn, type ChildProcess } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.resolve(__dirname, '../..')

const TRAIN_HTTP = process.env.VITE_TRAIN_API_URL ?? 'http://127.0.0.1:8787'
const WATCHDOG_INTERVAL_MS = 5_000

async function probeTrainService(): Promise<boolean> {
  try {
    const res = await fetch(`${TRAIN_HTTP}/api/train/status?env_id=`, {
      signal: AbortSignal.timeout(1500),
    })
    if (!res.ok) return false
    const body = (await res.json()) as { status?: string }
    return typeof body.status === 'string'
  } catch {
    return false
  }
}

function spawnTrainService(): ChildProcess {
  const proc = spawn('uv', ['run', '--project', 'swarm', 'python', '-m', 'swarm.train_service'], {
    cwd: REPO_ROOT,
    stdio: 'inherit',
    env: { ...process.env },
  })

  proc.on('error', (err) => {
    console.error(
      '[train-service] Failed to start (is `uv` installed?). Run manually:\n' +
        '  uv run --project swarm python -m swarm.train_service',
    )
    console.error(err.message)
  })

  proc.on('exit', (code) => {
    if (code !== 0 && code !== null) {
      console.warn(
        `[train-service] exited with code ${code}. If port 8787 is busy, stop the old process:\n` +
          '  lsof -ti :8787 | xargs kill',
      )
    }
  })

  return proc
}

export function trainServicePlugin(): Plugin {
  let proc: ChildProcess | null = null
  let startedByPlugin = false
  let watchdog: ReturnType<typeof setInterval> | null = null
  let shuttingDown = false

  const ensureRunning = async () => {
    if (shuttingDown) return
    if (proc && !proc.killed) return
    if (await probeTrainService()) return
    console.log('[train-service] (re)starting MAPPO bridge on :8787 / ws://localhost:8766 …')
    proc = spawnTrainService()
    startedByPlugin = true
  }

  const stop = () => {
    shuttingDown = true
    if (watchdog) {
      clearInterval(watchdog)
      watchdog = null
    }
    if (startedByPlugin && proc && !proc.killed) {
      proc.kill('SIGTERM')
      proc = null
      startedByPlugin = false
    }
  }

  return {
    name: 'combatos-train-service',
    apply: 'serve',
    async configureServer() {
      if (process.env.VITE_AUTO_TRAIN_SERVICE === '0') {
        console.log('[train-service] auto-start disabled (VITE_AUTO_TRAIN_SERVICE=0)')
        return
      }

      if (await probeTrainService()) {
        console.log(
          '[train-service] already running on :8787 — reusing (Train Policy should work)',
        )
      } else {
        await ensureRunning()
      }

      // Watchdog: respawn if the service dies after startup (handles crash-after-probe).
      if (!watchdog) {
        watchdog = setInterval(ensureRunning, WATCHDOG_INTERVAL_MS)
      }

      process.on('SIGINT', stop)
      process.on('SIGTERM', stop)
      process.on('exit', stop)
    },
    closeBundle() {
      stop()
    },
  }
}
