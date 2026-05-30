/**
 * verify_local_sim.mjs — headless parity check for the TS env port.
 *
 * Runs one full episode of the TypeScript SwarmEnv driven by the real trained
 * policies/search-and-interdict/policy.onnx (via onnxruntime-web in Node/Bun),
 * and prints final coverage.
 * If the obs construction / constants match env.py, the policy coordinates and
 * coverage reaches > 0.9 (Python reaches ~0.99). A wrong obs layout yields
 * garbage actions -> low coverage.
 *
 * Run from frontend/:
 *   bun scripts/verify_local_sim.mjs
 */
import * as ort from 'onnxruntime-web'
import { SwarmEnv, OBS_DIM, ACT_DIM } from '../src/swarm/sim.ts'

const ONNX = new URL(
  '../public/policies/search-and-interdict/policy.onnx',
  import.meta.url,
).pathname
const STEPS = 400
const KILL_AT = 200 // mirror the demo: kill an agent mid-rollout

ort.env.wasm.numThreads = 1

const session = await ort.InferenceSession.create(ONNX, {
  executionProviders: ['wasm'],
})
const inName = session.inputNames[0]
const outName = session.outputNames[0]

const env = new SwarmEnv(STEPS, 0)
let obs = env.observe()

for (let t = 0; t < STEPS; t++) {
  const tensor = new ort.Tensor('float32', obs, [env.n, OBS_DIM])
  const res = await session.run({ [inName]: tensor })
  const action = res[outName].data
  // action is n*ACT_DIM
  const a = new Float32Array(action.length)
  a.set(action)
  env.step(a)
  if (t === KILL_AT) env.kill(0)
  obs = env.observe()
}

const cov = env.coverage()
console.log(`steps=${STEPS}  killed agent 0 at step ${KILL_AT}`)
console.log(`final coverage = ${cov.toFixed(4)}  alive=${env.nAlive()}`)
console.log(`OBS_DIM=${OBS_DIM} ACT_DIM=${ACT_DIM}`)
if (cov > 0.9) {
  console.log('PARITY: PASS (coverage > 0.9)')
  process.exit(0)
} else {
  console.log('PARITY: FAIL (coverage <= 0.9 — obs/constants mismatch)')
  process.exit(1)
}
