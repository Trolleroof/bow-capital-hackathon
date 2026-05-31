/**
 * Optional backend autostart for local demos.
 *
 * Default workflow is now explicit:
 *   uv run --project swarm uvicorn swarm.backend:app --host 127.0.0.1 --port 8787
 *   bun dev
 *
 * Opt into autostart only when desired:
 *   VITE_AUTO_TRAIN_SERVICE=1 bun dev
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
  const proc = spawn('uv', ['run', '--project', 'swarm', 'uvicorn', 'swarm.backend:app', '--host', '127.0.0.1', '--port', '8787'], {
    cwd: REPO_ROOT,
    stdio: 'inherit',
    env: { ...process.env },
  })

  proc.on('error', (err) => {
    console.error(
      '[train-service] Failed to start (is `uv` installed?). Run manually:\n' +
        '  uv run --project swarm uvicorn swarm.backend:app --host 127.0.0.1 --port 8787',
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
    console.log('[backend] (re)starting API on :8787 …')
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
    name: 'outcast-virus-train-service',
    apply: 'serve',
    async configureServer() {
      if (process.env.VITE_AUTO_TRAIN_SERVICE !== '1') {
        console.log(
          '[backend] auto-start disabled. Run: uv run --project swarm uvicorn swarm.backend:app --host 127.0.0.1 --port 8787',
        )
        return
      }

      if (await probeTrainService()) {
        console.log(
          '[backend] already running on :8787 — reusing',
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
