import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { ALTITUDE, GRID, SwarmEnv, WORLD_HALF } from '../swarm/sim'
import { loadPolicy, type Policy } from '../swarm/policy'
import { makeDrone, updateDrone, Trail, type Drone } from '../swarm/drone'
import { drawMinimap, type MiniAgent } from '../swarm/minimap'

interface PoseSample {
  t: number
  x: number
  y: number
  z: number
  qx?: number
  qy?: number
  qz?: number
  qw?: number
}

interface CompositeScenePanelProps {
  trajectoryUrl?: string
  /**
   * Mock 3DGS point cloud by default. Swap-in seam for the REAL Gaussian splat
   * reconstructed from drone footage: point this at a JSON of {x,y,z,r,g,b}
   * (or replace `loadSplatPoints` with a true splat loader) and nothing else in
   * this scene — swarm, trajectory, coverage, controls — needs to change.
   */
  splatPointsUrl?: string
}

const FRAME_MS = 100
const CELL = (2 * WORLD_HALF) / GRID

// scene mapping: world (x,y,z) -> three (x = x, up = z, depth = -y)
function scenePoint(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, -y)
}

function fakeTrajectory(): PoseSample[] {
  const samples: PoseSample[] = []
  for (let i = 0; i < 180; i++) {
    const t = i / 20
    const a = i * 0.07
    samples.push({
      t,
      x: Math.cos(a) * (1.6 + i / 95),
      y: Math.sin(a) * (1.2 + i / 120),
      z: 0.7 + Math.sin(i * 0.08) * 0.35 + i / 170,
      qx: 0,
      qy: 0,
      qz: 0,
      qw: 1,
    })
  }
  return samples
}

// ----------------------------------------------------------------- splat ---
// (swap-in seam — see splatPointsUrl above)
function makeMockSplat() {
  const count = 2400
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    const r = Math.sqrt(Math.random()) * WORLD_HALF * 0.82
    const theta = Math.random() * Math.PI * 2
    const x = Math.cos(theta) * r
    const y = Math.sin(theta) * r
    const ridge = Math.sin(x * 0.8) * Math.cos(y * 0.55) * 0.45
    const z = -0.08 + ridge + Math.random() * 0.16
    const p = scenePoint(x, y, z)
    positions.set([p.x, p.y, p.z], i * 3)
    const warm = 0.28 + Math.random() * 0.26
    colors.set([warm, 0.38 + Math.random() * 0.28, 0.32 + Math.random() * 0.18], i * 3)
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  const material = new THREE.PointsMaterial({
    size: 0.095,
    vertexColors: true,
    transparent: true,
    opacity: 0.88,
    depthWrite: false,
  })
  return new THREE.Points(geometry, material)
}

async function loadTrajectory(url?: string): Promise<PoseSample[]> {
  if (!url) return fakeTrajectory()
  try {
    const res = await fetch(url)
    if (!res.ok) throw new Error(`${res.status}`)
    const data = await res.json()
    if (Array.isArray(data)) return data as PoseSample[]
    if (Array.isArray(data.frames)) return data.frames as PoseSample[]
  } catch {
    return fakeTrajectory()
  }
  return fakeTrajectory()
}

async function loadSplatPoints(url?: string): Promise<THREE.Points> {
  if (!url) return makeMockSplat()
  try {
    const res = await fetch(url)
    if (!res.ok) throw new Error(`${res.status}`)
    const data = await res.json()
    const points = Array.isArray(data) ? data : data.points
    if (!Array.isArray(points)) return makeMockSplat()
    const positions = new Float32Array(points.length * 3)
    const colors = new Float32Array(points.length * 3)
    for (let i = 0; i < points.length; i++) {
      const p = points[i]
      const v = scenePoint(Number(p.x) || 0, Number(p.y) || 0, Number(p.z) || 0)
      positions.set([v.x, v.y, v.z], i * 3)
      colors.set([Number(p.r) || 0.34, Number(p.g) || 0.54, Number(p.b) || 0.42], i * 3)
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    return new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ size: 0.09, vertexColors: true, transparent: true, opacity: 0.9 }),
    )
  } catch {
    return makeMockSplat()
  }
}

function heuristicActions(env: SwarmEnv): Float32Array {
  const actions = new Float32Array(env.n * 2)
  for (let i = 0; i < env.n; i++) {
    const angle = (i / env.n) * Math.PI * 2 + env.t * 0.18
    const targetX = Math.cos(angle) * WORLD_HALF * 0.74
    const targetY = Math.sin(angle) * WORLD_HALF * 0.74
    actions[i * 2] = Math.max(-1, Math.min(1, (targetX - env.pos[i * 2]) / 5))
    actions[i * 2 + 1] = Math.max(-1, Math.min(1, (targetY - env.pos[i * 2 + 1]) / 5))
  }
  return actions
}

const NOMINAL = 'all units nominal · coverage sweep active'

export default function CompositeScenePanel({ trajectoryUrl, splatPointsUrl }: CompositeScenePanelProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const miniRef = useRef<HTMLCanvasElement | null>(null)
  const envRef = useRef(new SwarmEnv(400, 7))
  const policyRef = useRef<Policy | null>(null)
  const killNextRef = useRef(false)
  const resetNextRef = useRef(false)
  const reviveNextRef = useRef(false)
  const autoKillRef = useRef(false)
  const [provider, setProvider] = useState('loading')
  const [alive, setAlive] = useState(envRef.current.nAlive())
  const [coverage, setCoverage] = useState(0)
  const [status, setStatus] = useState(NOMINAL)
  const [lost, setLost] = useState(false)
  const [autoKill, setAutoKill] = useState(false)

  // setStatus lives in React; the render loop signals via this ref.
  const statusRef = useRef<{ text: string; lost: boolean } | null>(null)

  useEffect(() => {
    let cancelled = false
    loadPolicy()
      .then((policy) => {
        if (!cancelled) {
          policyRef.current = policy
          setProvider(policy.provider)
        }
      })
      .catch(() => {
        if (!cancelled) setProvider('heuristic')
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const width = mount.clientWidth || 960
    const height = mount.clientHeight || 640
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x05080a)
    scene.fog = new THREE.FogExp2(0x05080a, 0.034)

    const camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 1000)
    camera.position.set(7.4, 6.1, 8.2)
    camera.lookAt(0, 1.1, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(width, height)
    mount.appendChild(renderer.domElement)

    scene.add(new THREE.HemisphereLight(0xc9f0ff, 0x111911, 1.45))
    const key = new THREE.DirectionalLight(0xffffff, 1.4)
    key.position.set(4, 10, 6)
    scene.add(key)

    const grid = new THREE.GridHelper(WORLD_HALF * 2, GRID, 0x365547, 0x16231f)
    scene.add(grid)

    // --- coverage paint: one instanced quad per grid cell, shown as covered ---
    const env0 = envRef.current
    const cellGeo = new THREE.PlaneGeometry(CELL * 0.92, CELL * 0.92)
    cellGeo.rotateX(-Math.PI / 2)
    const cellMat = new THREE.MeshBasicMaterial({
      color: 0x2fae7a,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
    })
    const coverMesh = new THREE.InstancedMesh(cellGeo, cellMat, GRID * GRID)
    coverMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    scene.add(coverMesh)
    const dummy = new THREE.Object3D()
    const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0)
    for (let idx = 0; idx < GRID * GRID; idx++) coverMesh.setMatrixAt(idx, HIDDEN)
    coverMesh.instanceMatrix.needsUpdate = true
    let lastCoveredCount = -1
    const refreshCoverage = () => {
      const covered = env0.coveredCells()
      let count = 0
      for (let cx = 0; cx < GRID; cx++) {
        for (let cy = 0; cy < GRID; cy++) {
          const idx = cx * GRID + cy
          if (covered[idx]) {
            count++
            const wx = (cx + 0.5) * CELL - WORLD_HALF
            const wy = (cy + 0.5) * CELL - WORLD_HALF
            dummy.position.set(wx, 0.02, -wy)
            dummy.scale.setScalar(1)
            dummy.updateMatrix()
            coverMesh.setMatrixAt(idx, dummy.matrix)
          }
        }
      }
      if (count !== lastCoveredCount) {
        coverMesh.instanceMatrix.needsUpdate = true
        lastCoveredCount = count
      }
    }

    const trajectoryLine = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x55b8ff, transparent: true, opacity: 0.95 }),
    )
    scene.add(trajectoryLine)

    const cameraMarker = new THREE.Mesh(
      new THREE.ConeGeometry(0.18, 0.42, 18),
      new THREE.MeshStandardMaterial({ color: 0x55b8ff, emissive: 0x06233d }),
    )
    scene.add(cameraMarker)

    // --- drones (quadrotor meshes + trails from drone.ts) ---
    const drones: Drone[] = Array.from({ length: env0.n }, () => makeDrone())
    drones.forEach((d) => scene.add(d.group))
    const current: THREE.Vector3[] = drones.map((_, i) =>
      scenePoint(env0.pos[i * 2], env0.pos[i * 2 + 1], ALTITUDE),
    )
    const trails = drones.map(() => {
      const t = new Trail(0x4df09a)
      scene.add(t.line)
      return t
    })

    let poses = fakeTrajectory()
    let frameTimer = 0
    let disposed = false
    let stepping = false
    void loadTrajectory(trajectoryUrl).then((loaded) => {
      poses = loaded
      const vertices = poses.map((p) => scenePoint(p.x, p.y, p.z))
      trajectoryLine.geometry.dispose()
      trajectoryLine.geometry = new THREE.BufferGeometry().setFromPoints(vertices)
    })
    void loadSplatPoints(splatPointsUrl).then((points) => {
      if (!disposed) scene.add(points)
    })

    // --- click-to-kill: raycast against drone groups -------------------------
    const raycaster = new THREE.Raycaster()
    const ndc = new THREE.Vector2()
    const onPointerDown = (ev: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1
      ndc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(ndc, camera)
      const hits = raycaster.intersectObjects(
        drones.map((d) => d.group),
        true,
      )
      if (!hits.length) return
      // walk up to the top-level drone group
      let obj: THREE.Object3D | null = hits[0].object
      while (obj && !drones.some((d) => d.group === obj)) obj = obj.parent
      if (!obj) return
      const i = drones.findIndex((d) => d.group === obj)
      if (i >= 0 && envRef.current.alive[i]) {
        envRef.current.kill(i)
        statusRef.current = { text: `agent ${i} lost — swarm recovering, redistributing coverage`, lost: true }
      }
    }
    renderer.domElement.addEventListener('pointerdown', onPointerDown)

    let raf = 0
    let last = performance.now()
    const animate = (now: number) => {
      raf = requestAnimationFrame(animate)
      const delta = now - last
      last = now
      const dt = delta / 1000
      frameTimer += delta

      const env = envRef.current

      if (resetNextRef.current) {
        resetNextRef.current = false
        env.reset(Math.floor(Math.random() * 1e9))
        trails.forEach((t) => t.clear())
        lastCoveredCount = -1
        for (let idx = 0; idx < GRID * GRID; idx++) coverMesh.setMatrixAt(idx, HIDDEN)
        coverMesh.instanceMatrix.needsUpdate = true
        statusRef.current = { text: NOMINAL, lost: false }
      }
      if (reviveNextRef.current) {
        reviveNextRef.current = false
        env.reviveAll()
        statusRef.current = { text: NOMINAL, lost: false }
      }
      if (killNextRef.current) {
        killNextRef.current = false
        const lastAlive = env.alive.findLastIndex(Boolean)
        if (lastAlive >= 0) {
          env.kill(lastAlive)
          statusRef.current = { text: `agent ${lastAlive} lost — swarm recovering, redistributing coverage`, lost: true }
        }
      }

      if (frameTimer >= FRAME_MS && !stepping) {
        frameTimer = 0
        stepping = true
        // optional auto-kill at step 200 (off by default)
        if (autoKillRef.current && env.steps === 200 && env.nAlive() === env.n) {
          env.kill(0)
          statusRef.current = { text: 'agent 0 lost — swarm recovering, redistributing coverage', lost: true }
        }
        const obs = env.observe()
        const actionPromise = policyRef.current
          ? policyRef.current.act(obs, env.n)
          : Promise.resolve(heuristicActions(env))
        void actionPromise
          .catch(() => {
            setProvider('heuristic')
            return heuristicActions(env)
          })
          .then((actions) => {
            env.step(actions)
            setAlive(env.nAlive())
            setCoverage(env.coverage())
            refreshCoverage()
            stepping = false
          })
      }

      // drones: smooth toward target, then drone.ts handles spin + dead-state.
      // updateDrone(x, y, z) places at position.set(x, z, y), so we pass the
      // scene vector as (sx, sz, sy) to land at the intended (sx, sy, sz).
      for (let i = 0; i < env.n; i++) {
        const target = scenePoint(
          env.pos[i * 2],
          env.pos[i * 2 + 1],
          ALTITUDE + Math.sin(now * 0.002 + i) * 0.18,
        )
        current[i].lerp(target, 0.26)
        const yaw = -Math.atan2(env.vel[i * 2 + 1], env.vel[i * 2])
        updateDrone(drones[i], current[i].x, current[i].z, current[i].y, yaw, env.alive[i], dt)
        if (env.alive[i]) trails[i].push(current[i].x, current[i].z, current[i].y)
      }

      // minimap (top-down coordination map)
      const mini = miniRef.current
      if (mini) {
        const ctx = mini.getContext('2d')
        if (ctx) {
          const agents: MiniAgent[] = []
          for (let i = 0; i < env.n; i++) {
            agents.push({ x: env.pos[i * 2], y: env.pos[i * 2 + 1], alive: env.alive[i] })
          }
          drawMinimap(ctx, mini.width, WORLD_HALF, GRID, env.coveredCells(), agents)
        }
      }

      // status line (signaled from render loop / raycast handler)
      if (statusRef.current) {
        setStatus(statusRef.current.text)
        setLost(statusRef.current.lost)
        statusRef.current = null
      }

      const pose = poses[Math.floor((now * 0.018) % poses.length)]
      if (pose) {
        cameraMarker.position.copy(scenePoint(pose.x, pose.y, pose.z))
        cameraMarker.rotation.y = now * 0.0018
      }
      camera.position.x = Math.cos(now * 0.00012) * 8.4
      camera.position.z = Math.sin(now * 0.00012) * 8.4
      camera.lookAt(0, 1.1, 0)
      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(animate)

    const onResize = () => {
      const w = mount.clientWidth || width
      const h = mount.clientHeight || height
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.dispose()
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Points || obj instanceof THREE.Line) {
          obj.geometry.dispose()
          const material = obj.material
          if (Array.isArray(material)) material.forEach((m) => m.dispose())
          else material.dispose()
        }
      })
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement)
    }
  }, [splatPointsUrl, trajectoryUrl])

  return (
    <section className="composite-scene" style={{ position: 'relative' }}>
      <div ref={mountRef} className="composite-canvas" style={{ cursor: 'crosshair' }} />

      {/* coordination minimap (top-right) */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          padding: 6,
          background: 'rgba(7,18,14,0.78)',
          border: '1px solid rgba(47,174,122,0.5)',
          borderRadius: 4,
          fontFamily: 'monospace',
          color: '#4ef0a0',
        }}
      >
        <div style={{ fontSize: 10, letterSpacing: 1, opacity: 0.8, marginBottom: 4 }}>
          COORDINATION MAP
        </div>
        <canvas ref={miniRef} width={150} height={150} style={{ display: 'block' }} />
      </div>

      {/* event status line */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '4px 12px',
          fontFamily: 'monospace',
          fontSize: 13,
          color: lost ? '#ff6b6b' : '#4ef0a0',
          background: 'rgba(5,8,10,0.7)',
          border: `1px solid ${lost ? 'rgba(255,107,107,0.5)' : 'rgba(47,174,122,0.4)'}`,
          borderRadius: 3,
          whiteSpace: 'nowrap',
        }}
      >
        {lost ? '▸ ' : '▸ '}
        {status}
      </div>

      <div className="mission-strip">
        <div>
          <strong>GPS: DENIED</strong>
          <span>LINK: NONE</span>
          <span>LOCALIZED</span>
        </div>
        <div>
          <span>POLICY: {provider.toUpperCase()}</span>
          <span>ALIVE: {alive}</span>
          <span>COVERAGE: {Math.round(coverage * 100)}%</span>
        </div>
      </div>

      <div className="scene-actions">
        <button
          type="button"
          style={{ borderColor: '#ff6b6b', color: '#ff6b6b' }}
          onClick={() => {
            killNextRef.current = true
          }}
        >
          Kill Agent
        </button>
        <button type="button" onClick={() => { reviveNextRef.current = true }}>
          Revive All
        </button>
        <button type="button" onClick={() => { resetNextRef.current = true }}>
          Reset
        </button>
        <label style={{ fontFamily: 'monospace', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={autoKill}
            onChange={(e) => {
              setAutoKill(e.target.checked)
              autoKillRef.current = e.target.checked
            }}
          />
          auto-kill @200
        </label>
        <span style={{ fontFamily: 'monospace', fontSize: 11, opacity: 0.6 }}>
          click a drone to kill it
        </span>
      </div>
    </section>
  )
}
