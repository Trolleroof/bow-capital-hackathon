/**
 * SwarmPanel — Three.js view of the CombatOS swarm.
 *
 * Two data sources (prop `source`):
 *   'local' (default) — Phase 3 / Edge A: runs the trained policy.onnx CLIENT-SIDE
 *      via onnxruntime-web (WASM), stepping a faithful TS port of SwarmEnv each
 *      frame. No WebSocket, no Python — works fully offline.
 *   'bus' — Phase 0 fallback: subscribes to the Python WebSocket bus (swarm/bus.py)
 *      `swarm` topic and renders the streamed agents.
 *
 * Phase 4 adds an ops-console layer on top of the in-browser inference:
 *   - quadrotor drone meshes + fading motion trails
 *   - the coverage grid tinted on the ground (watch the swarm paint the field)
 *   - a top-down "COORDINATION MAP" minimap
 *   - operator controls: REVIVE and RESET
 *   - an event status line ("agent 3 lost — swarm recovering")
 *
 * Bus message (SWARM.md §4 / TEAM_PLAN §5):
 *   { "topic": "swarm", "t": 1234.56, "comms": "denied",
 *     "agents": [ { id, x, y, z, yaw, role, alive }, ... ] }
 */
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { SwarmEnv, WORLD_HALF as ENV_WORLD_HALF, ALTITUDE, GRID } from '../swarm/sim'
import { loadPolicy, type Policy } from '../swarm/policy'
import { makeDrone, updateDrone, Trail, type Drone } from '../swarm/drone'
import { drawMinimap, type MiniAgent } from '../swarm/minimap'

// --- bus message types -------------------------------------------------------
interface SwarmAgent {
  id: number
  x: number
  y: number
  z: number
  yaw: number
  role: string
  alive: boolean
}
interface SwarmMessage {
  topic?: string
  t: number
  comms: string
  agents: SwarmAgent[]
}

// World spans [-WORLD_HALF, WORLD_HALF] in x and y (matches env.WORLD_HALF).
const WORLD_HALF = ENV_WORLD_HALF
const TRAINED_COVERAGE = 0.9965
const RANDOM_COVERAGE = 0.4725

interface SwarmPanelProps {
  url?: string
  /** 'local' = browser onnx inference (default), 'bus' = WebSocket fallback. */
  source?: 'bus' | 'local'
}

export default function SwarmPanel({
  url = 'ws://localhost:8765',
  source = 'local',
}: SwarmPanelProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const miniRef = useRef<HTMLCanvasElement | null>(null)
  // latest bus message, written by the WS handler and read by the render loop
  const latest = useRef<SwarmMessage | null>(null)
  const [connected, setConnected] = useState(false)
  const [nAlive, setNAlive] = useState(0)
  const [comms, setComms] = useState(source === 'local' ? 'denied' : '—')
  const [coverage, setCoverage] = useState(0)
  const [edgeLabel, setEdgeLabel] = useState(
    source === 'local' ? 'BROWSER (onnx … loading)' : 'bus',
  )
  const [policyState, setPolicyState] = useState<'loading' | 'active' | 'failed' | 'bus'>(
    source === 'local' ? 'loading' : 'bus',
  )
  const [meanAction, setMeanAction] = useState(0)
  const [status, setStatus] = useState('all units nominal · coverage sweep active')

  // --- WebSocket subscription (bus mode only) ---
  useEffect(() => {
    if (source !== 'bus') return
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let closed = false

    const connect = () => {
      ws = new WebSocket(url)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = setTimeout(connect, 1000)
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as SwarmMessage
          if (msg.topic && msg.topic !== 'swarm') return
          if (!Array.isArray(msg.agents)) return
          latest.current = msg
          setComms(msg.comms ?? '—')
          setNAlive(msg.agents.filter((a) => a.alive).length)
        } catch {
          /* ignore malformed frames */
        }
      }
    }
    connect()
    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      ws?.close()
    }
  }, [url, source])

  // --- local browser inference (local mode only): load policy + step sim ---
  // Holds the sim + policy so operator handlers (kill/revive/reset) can reach
  // them via the ref.
  const simRef = useRef<SwarmEnv | null>(null)
  const policyRef = useRef<Policy | null>(null)

  useEffect(() => {
    if (source !== 'local') return
    let cancelled = false
    const sim = new SwarmEnv(/* maxSteps */ 100000, /* seed */ 0)
    simRef.current = sim

    loadPolicy('search-and-interdict')
      .then((p) => {
        if (cancelled) return
        policyRef.current = p
        setEdgeLabel(`BROWSER (onnx · ${p.provider})`)
        setPolicyState('active')
        setStatus('trained policy active · coverage objective executing')
      })
      .catch((err) => {
        console.error('[swarm] failed to load policy.onnx:', err)
        setEdgeLabel('BROWSER (onnx FAILED)')
        setPolicyState('failed')
        setStatus('policy load failed · no learned control available')
      })

    return () => {
      cancelled = true
      simRef.current = null
      policyRef.current = null
    }
  }, [source])

  // --- operator actions (local mode) ---
  const reviveAll = () => {
    const sim = simRef.current
    if (!sim) return
    sim.reviveAll()
    setNAlive(sim.nAlive())
    setStatus('all units restored · swarm re-formed')
  }

  const resetEpisode = () => {
    const sim = simRef.current
    if (!sim) return
    sim.reset(Math.floor(Math.random() * 1e9))
    setNAlive(sim.nAlive())
    setCoverage(sim.coverage())
    setStatus('episode reset · coverage sweep restarting')
    clearTrailsRef.current?.()
  }

  // lets resetEpisode wipe the trails created inside the three.js effect
  const clearTrailsRef = useRef<(() => void) | null>(null)

  // --- Three.js scene (both modes) ---
  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const width = mount.clientWidth || 480
    const height = mount.clientHeight || 360

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0e12)

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000)
    camera.position.set(0, WORLD_HALF * 2.2, WORLD_HALF * 1.6)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(width, height)
    mount.appendChild(renderer.domElement)

    scene.add(new THREE.AmbientLight(0xffffff, 0.7))
    const dir = new THREE.DirectionalLight(0xffffff, 0.7)
    dir.position.set(5, 20, 10)
    scene.add(dir)

    // bounded world: ground grid + edge box
    const grid = new THREE.GridHelper(WORLD_HALF * 2, GRID, 0x1f6f4f, 0x16313a)
    scene.add(grid)
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(
        new THREE.BoxGeometry(WORLD_HALF * 2, 6, WORLD_HALF * 2),
      ),
      new THREE.LineBasicMaterial({ color: 0x2fae7a }),
    )
    box.position.y = 3
    scene.add(box)

    // --- coverage ground: one InstancedMesh of GRID*GRID flat quads, faded in
    // as cells get covered (alpha via per-instance scale + emissive material). ---
    const cellSize = (WORLD_HALF * 2) / GRID
    const coverGeo = new THREE.PlaneGeometry(cellSize * 0.96, cellSize * 0.96)
    coverGeo.rotateX(-Math.PI / 2)
    const coverMat = new THREE.MeshBasicMaterial({
      color: 0x4ef0a0,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
    })
    const coverMesh = new THREE.InstancedMesh(coverGeo, coverMat, GRID * GRID)
    coverMesh.position.y = 0.02
    coverMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    scene.add(coverMesh)
    const hideM = new THREE.Matrix4().makeScale(0, 0, 0)
    const cellM = new THREE.Matrix4()
    const tmpPos = new THREE.Vector3()
    // env cell (cx,cy): world x = -WORLD_HALF + (cx+0.5)*cellSize, world y same
    // for cy. scene maps (x,y) -> (x, _, y).
    const coverShown = new Uint8Array(GRID * GRID)
    for (let i = 0; i < GRID * GRID; i++) coverMesh.setMatrixAt(i, hideM)
    coverMesh.instanceMatrix.needsUpdate = true

    const refreshCover = (covered: Uint8Array) => {
      let changed = false
      for (let cx = 0; cx < GRID; cx++) {
        for (let cy = 0; cy < GRID; cy++) {
          const gi = cx * GRID + cy
          if (covered[gi] && !coverShown[gi]) {
            const wx = -WORLD_HALF + (cx + 0.5) * cellSize
            const wy = -WORLD_HALF + (cy + 0.5) * cellSize
            tmpPos.set(wx, 0, wy)
            cellM.makeTranslation(tmpPos.x, tmpPos.y, tmpPos.z)
            coverMesh.setMatrixAt(gi, cellM)
            coverShown[gi] = 1
            changed = true
          } else if (!covered[gi] && coverShown[gi]) {
            coverMesh.setMatrixAt(gi, hideM)
            coverShown[gi] = 0
            changed = true
          }
        }
      }
      if (changed) coverMesh.instanceMatrix.needsUpdate = true
    }

    // drone meshes + trails, created lazily / reused across frames
    const drones: Drone[] = []
    const trails: Trail[] = []
    const actionArrows: THREE.ArrowHelper[] = []
    const ensureDrone = (i: number): Drone => {
      let d = drones[i]
      if (!d) {
        d = makeDrone()
        drones[i] = d
        scene.add(d.group)
        const t = new Trail(0x2fae7a)
        trails[i] = t
        scene.add(t.line)
        const arrow = new THREE.ArrowHelper(
          new THREE.Vector3(1, 0, 0),
          new THREE.Vector3(0, ALTITUDE + 0.25, 0),
          1.8,
          0x7fd9ff,
          0.45,
          0.22,
        )
        actionArrows[i] = arrow
        scene.add(arrow)
      }
      return d
    }

    clearTrailsRef.current = () => {
      for (const t of trails) t.clear()
      for (let i = 0; i < GRID * GRID; i++) coverMesh.setMatrixAt(i, hideM)
      coverShown.fill(0)
      coverMesh.instanceMatrix.needsUpdate = true
    }

    // Coord mapping note: env (x,y,z) -> three (x, z, -y); env y is the ground
    // plane, env z (= ALTITUDE) is three's vertical axis. updateDrone takes
    // already-mapped args, so we pass (x, ALTITUDE, -y).

    renderer.domElement.style.cursor = 'default'

    const miniCtx = miniRef.current?.getContext('2d') ?? null
    const miniSize = miniRef.current?.width ?? 0

    // local-mode async inference guard
    let inferring = false
    let last = performance.now()

    let raf = 0
    const animate = () => {
      raf = requestAnimationFrame(animate)
      const now = performance.now()
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now

      if (source === 'local') {
        const sim = simRef.current
        const policy = policyRef.current
        if (sim && policy && !inferring) {
          inferring = true
          const obs = sim.observe()
          policy
            .act(obs, sim.n)
            .then((actions) => {
              sim.step(actions)
              setNAlive(sim.nAlive())
              setCoverage(sim.coverage())
              let total = 0
              for (let i = 0; i < sim.n; i++) {
                const ax = actions[i * 2]
                const ay = actions[i * 2 + 1]
                total += Math.hypot(ax, ay)
              }
              setMeanAction(total / sim.n)
            })
            .catch((err) => {
              console.error('[swarm] inference error:', err)
            })
            .finally(() => {
              inferring = false
            })
        }
        if (sim) {
          refreshCover(sim.coveredCells())
          for (let i = 0; i < sim.n; i++) {
            const x = sim.pos[i * 2]
            const y = sim.pos[i * 2 + 1]
            const alive = sim.alive[i]
            const yaw = Math.atan2(sim.vel[i * 2], sim.vel[i * 2 + 1])
            const d = ensureDrone(i)
            updateDrone(d, x, ALTITUDE, -y, yaw, alive, dt)
            if (alive) trails[i].push(x, -y, ALTITUDE)
            const arrow = actionArrows[i]
            if (arrow) {
              const ax = sim.vel[i * 2]
              const ay = sim.vel[i * 2 + 1]
              const speed = Math.hypot(ax, ay)
              arrow.visible = alive && speed > 0.03
              arrow.position.set(x, ALTITUDE + 0.45, -y)
              if (speed > 0.03) {
                arrow.setDirection(new THREE.Vector3(ax, 0, -ay).normalize())
                arrow.setLength(1.0 + speed * 1.4, 0.45, 0.22)
              }
            }
          }
          if (miniCtx) {
            const agents: MiniAgent[] = []
            for (let i = 0; i < sim.n; i++) {
              agents.push({ x: sim.pos[i * 2], y: sim.pos[i * 2 + 1], alive: sim.alive[i] })
            }
            drawMinimap(miniCtx, miniSize, WORLD_HALF, GRID, sim.coveredCells(), agents)
          }
        }
      } else {
        const msg = latest.current
        if (msg) {
          for (let i = 0; i < msg.agents.length; i++) {
            const a = msg.agents[i]
            const d = ensureDrone(i)
            const yaw = a.yaw ?? 0
            updateDrone(d, a.x, a.z, -a.y, yaw, a.alive, dt)
            if (a.alive) trails[i].push(a.x, -a.y, a.z)
          }
          for (let i = msg.agents.length; i < drones.length; i++) {
            drones[i].group.visible = false
          }
          for (const arrow of actionArrows) arrow.visible = false
          if (miniCtx) {
            const agents: MiniAgent[] = msg.agents.map((a) => ({
              x: a.x,
              y: a.y,
              alive: a.alive,
            }))
            drawMinimap(miniCtx, miniSize, WORLD_HALF, GRID, new Uint8Array(GRID * GRID), agents)
          }
        }
      }

      renderer.render(scene, camera)
    }
    animate()

    const onResize = () => {
      const w = mount.clientWidth || width
      const h = mount.clientHeight || height
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      clearTrailsRef.current = null
      for (const t of trails) t.dispose()
      for (const arrow of actionArrows) {
        arrow.line.geometry.dispose()
        arrow.cone.geometry.dispose()
      }
      coverGeo.dispose()
      coverMat.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement)
      }
    }
  }, [source])

  const local = source === 'local'
  const displayComms = local ? 'denied' : comms

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 360 }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* --- HUD (top-left) --- */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          fontFamily: 'monospace',
          color: '#4ef0a0',
          pointerEvents: 'none',
          textShadow: '0 0 4px #000',
          lineHeight: 1.5,
        }}
      >
        <div style={{ fontWeight: 700, letterSpacing: 1 }}>
          COMMS: {displayComms.toUpperCase()}
        </div>
        <div style={{ opacity: 0.85 }}>AGENTS ALIVE: {nAlive}</div>
        <div style={{ opacity: 0.85 }}>EDGE: {local ? edgeLabel : 'bus'}</div>
        {local && (
          <div
            style={{
              opacity: 0.95,
              color: policyState === 'active' ? '#7fd9ff' : policyState === 'failed' ? '#ff6b6b' : '#d5b76a',
            }}
          >
            POLICY: {policyState.toUpperCase()}
          </div>
        )}
        {local && policyState === 'active' && (
          <>
            <div style={{ opacity: 0.85 }}>
              EVAL: {(TRAINED_COVERAGE * 100).toFixed(0)}% / RANDOM {(RANDOM_COVERAGE * 100).toFixed(0)}%
            </div>
            <div style={{ opacity: 0.85 }}>
              CMD: {(meanAction * 100).toFixed(0)}%
            </div>
          </>
        )}
        {local && (
          <div style={{ opacity: 0.85 }}>
            COVERAGE: {(coverage * 100).toFixed(0)}%
          </div>
        )}
        <div style={{ opacity: 0.6, fontSize: 11 }}>
          {local
            ? 'client-side · offline'
            : connected
              ? 'bus connected'
              : 'connecting…'}
        </div>
      </div>

      {/* --- event status line (top-center) --- */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: '50%',
          transform: 'translateX(-50%)',
          fontFamily: 'monospace',
          fontSize: 12,
          color: status.includes('lost') ? '#ff6b6b' : '#7fd9b0',
          background: 'rgba(6, 12, 10, 0.6)',
          border: `1px solid ${status.includes('lost') ? '#5a2020' : '#16313a'}`,
          padding: '4px 10px',
          borderRadius: 3,
          pointerEvents: 'none',
          letterSpacing: 0.5,
          maxWidth: '70%',
          textAlign: 'center',
        }}
      >
        ▸ {status}
      </div>

      {/* --- coordination minimap (top-right) --- */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          fontFamily: 'monospace',
          color: '#4ef0a0',
          background: 'rgba(6, 12, 10, 0.55)',
          border: '1px solid #16313a',
          borderRadius: 4,
          padding: 6,
          pointerEvents: 'none',
        }}
      >
        <div style={{ fontSize: 10, letterSpacing: 1, opacity: 0.8, marginBottom: 4 }}>
          COORDINATION MAP
        </div>
        <canvas
          ref={miniRef}
          width={160}
          height={160}
          style={{ display: 'block', width: 160, height: 160, borderRadius: 2 }}
        />
      </div>

      {/* --- operator controls (bottom-center) --- */}
      {local && (
        <div
          style={{
            position: 'absolute',
            bottom: 16,
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            fontFamily: 'monospace',
            background: 'rgba(6, 12, 10, 0.7)',
            border: '1px solid #16313a',
            borderRadius: 6,
            padding: '8px 12px',
            backdropFilter: 'blur(2px)',
          }}
        >
          <span style={{ color: '#4ef0a0', fontSize: 10, letterSpacing: 1, opacity: 0.7 }}>
            OPERATOR
          </span>
          <button onClick={reviveAll} style={btnStyle}>
            ⟲ REVIVE ALL
          </button>
          <button onClick={resetEpisode} style={btnStyle}>
            ⟳ RESET
          </button>
        </div>
      )}
    </div>
  )
}

const btnStyle: React.CSSProperties = {
  fontFamily: 'monospace',
  fontSize: 11,
  letterSpacing: 1,
  color: '#4ef0a0',
  background: '#0c1f18',
  border: '1px solid #2fae7a',
  borderRadius: 4,
  padding: '6px 12px',
  cursor: 'pointer',
}
