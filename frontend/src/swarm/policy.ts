/**
 * policy.ts — onnxruntime-web wrapper for the trained MAPPO actor.
 *
 * Loads swarm/checkpoints/<envId>/policy.onnx and runs inference client-side
 * (WASM execution provider) — no Python, works offline.
 *
 * ONNX contract (from swarm/export_onnx.py):
 *   input  "obs"     float32  (N, 36)   dynamic axis 0 = batch
 *   output "action"  float32  (N, 2)    deterministic, in [-1, 1]
 */
import * as ort from 'onnxruntime-web'
import { OBS_DIM, ACT_DIM } from './sim'

export interface Policy {
  /** Run inference: obs is N*36 flat float32, returns N*2 flat float32. */
  act(obs: Float32Array, n: number): Promise<Float32Array>
  /** Which ORT execution provider actually loaded (for the HUD). */
  readonly provider: string
}

/** Lifecycle state of a trained checkpoint for a given environment. */
export type PolicyStatus = 'not-trained' | 'training' | 'ready'

/**
 * Module-level active env ID.  Set this before Mission Sim mounts so that
 * CompositeScenePanel's zero-arg `loadPolicy()` call resolves to the correct
 * per-environment checkpoint without requiring a prop change.
 */
let _activeEnvId: string | null = null

/** Call from App before rendering Mission Sim to bind the checkpoint path. */
export function setActiveEnvId(envId: string | null): void {
  _activeEnvId = envId
}

/**
 * Probe whether a trained checkpoint exists for `envId`.
 * Uses HEAD so it doesn't download the (potentially large) ONNX blob.
 */
export async function checkPolicyExists(envId: string): Promise<boolean> {
  try {
    const res = await fetch(`/swarm/checkpoints/${envId}/policy.onnx`, {
      method: 'HEAD',
    })
    return res.ok
  } catch {
    return false
  }
}

let wasmConfigured = false

/**
 * Configure ORT's wasm asset paths for Vite. onnxruntime-web ships its .wasm
 * files in node_modules/onnxruntime-web/dist; Vite serves them via the bundled
 * URL helper below, so there are no 404s in dev or in the production build.
 */
function configureWasm() {
  if (wasmConfigured) return
  wasmConfigured = true
  // Map the ORT wasm/mjs assets to Vite-resolved URLs (works in dev + build).
  // The single-threaded SIMD build is the most portable (no COOP/COEP headers
  // needed), so we pin it explicitly.
  ort.env.wasm.numThreads = 1
  ort.env.wasm.wasmPaths = {
    wasm: new URL(
      '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm',
      import.meta.url,
    ).href,
    mjs: new URL(
      '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs',
      import.meta.url,
    ).href,
  }
}

/**
 * Load the policy and return an `act()` runner.
 *
 * URL resolution order:
 *  1. Explicit `url` argument (legacy / test override)
 *  2. `/swarm/checkpoints/<_activeEnvId>/policy.onnx` when an env is active
 *  3. `/policy.onnx` (legacy fallback)
 */
export async function loadPolicy(url?: string): Promise<Policy> {
  const resolvedUrl =
    url ??
    (_activeEnvId
      ? `/swarm/checkpoints/${_activeEnvId}/policy.onnx`
      : '/policy.onnx')

  configureWasm()

  const session = await ort.InferenceSession.create(resolvedUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })

  const inputName = session.inputNames[0] ?? 'obs'
  const outputName = session.outputNames[0] ?? 'action'

  return {
    provider: 'wasm',
    async act(obs: Float32Array, n: number): Promise<Float32Array> {
      if (obs.length !== n * OBS_DIM) {
        throw new Error(
          `obs length ${obs.length} != n*${OBS_DIM} (${n * OBS_DIM})`,
        )
      }
      const tensor = new ort.Tensor('float32', obs, [n, OBS_DIM])
      const out = await session.run({ [inputName]: tensor })
      const action = out[outputName]
      const data = action.data as Float32Array
      // expect n*ACT_DIM
      if (data.length !== n * ACT_DIM) {
        throw new Error(`action length ${data.length} != n*${ACT_DIM}`)
      }
      // return a copy so callers can hold it across frames safely
      return new Float32Array(data)
    },
  }
}
