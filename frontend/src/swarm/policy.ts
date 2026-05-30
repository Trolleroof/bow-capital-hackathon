/**
 * policy.ts — onnxruntime-web wrapper for the trained MAPPO actor.
 *
 * Loads frontend/public/policy.onnx and runs it client-side (WASM execution
 * provider) so the swarm's neural net inference happens entirely in the
 * browser — no Python, works offline.
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

/** Load the policy and return an `act()` runner. */
export async function loadPolicy(url = '/policy.onnx'): Promise<Policy> {
  configureWasm()

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
      // expect n*ACT_DIM
      if (data.length !== n * ACT_DIM) {
        throw new Error(`action length ${data.length} != n*${ACT_DIM}`)
      }
      // return a copy so callers can hold it across frames safely
      return new Float32Array(data)
    },
  }
}
