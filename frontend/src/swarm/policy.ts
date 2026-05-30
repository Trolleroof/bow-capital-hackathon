/**
 * policy.ts — onnxruntime-web wrapper for the trained MAPPO actor.
 *
 * Loads frontend/public/policies/<envId>/policy.onnx (browser-served copy of
 * swarm/checkpoints/<envId>/policy.onnx from export_onnx.py) and runs inference
 * client-side (WASM) — no Python, works offline.
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

const DEFAULT_ENV_ID = 'search-and-interdict'

/**
 * Module-level active env ID. Set before Mission Sim mounts so
 * CompositeScenePanel's zero-arg `loadPolicy()` resolves correctly (#23).
 */
let _activeEnvId: string | null = null

/** Call from App before rendering Mission Sim to bind the checkpoint path. */
export function setActiveEnvId(envId: string | null): void {
  _activeEnvId = envId
}

function resolvePolicyPath(envIdOrUrl?: string): string {
  const target = envIdOrUrl ?? _activeEnvId ?? DEFAULT_ENV_ID
  if (
    target.startsWith('/') ||
    target.startsWith('./') ||
    target.startsWith('../') ||
    target.endsWith('.onnx')
  ) {
    return target
  }
  return `/policies/${target}/policy.onnx`
}

/**
 * Probe whether a trained checkpoint exists for `envId`.
 * Uses HEAD so it doesn't download the ONNX blob.
 */
export async function checkPolicyExists(envId: string): Promise<boolean> {
  try {
    const res = await fetch(resolvePolicyPath(envId), { method: 'HEAD' })
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
 * Resolution order:
 *  1. Explicit env id or full URL/path argument
 *  2. `_activeEnvId` set via setActiveEnvId()
 *  3. DEFAULT_ENV_ID (`search-and-interdict`)
 */
export async function loadPolicy(envIdOrUrl?: string): Promise<Policy> {
  configureWasm()
  const url = resolvePolicyPath(envIdOrUrl)

  const session = await ort.InferenceSession.create(url, {
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
      if (data.length !== n * ACT_DIM) {
        throw new Error(`action length ${data.length} != n*${ACT_DIM}`)
      }
      return new Float32Array(data)
    },
  }
}
