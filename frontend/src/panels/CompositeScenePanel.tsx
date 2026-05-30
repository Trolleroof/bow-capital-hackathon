import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { ALTITUDE, GRID, N_AGENTS, SwarmEnv, WORLD_HALF, MAX_SPEED } from '../swarm/sim'
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
  missionName?: string
  missionBrief?: string
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
    const soil = 0.3 + Math.random() * 0.18
    colors.set([soil, 0.44 + Math.random() * 0.22, 0.24 + Math.random() * 0.08], i * 3)
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

export default function CompositeScenePanel({
  trajectoryUrl,
  splatPointsUrl,
  missionName = 'Land Coverage Survey',
  missionBrief = 'Field reconstruction and coverage sweep',
}: CompositeScenePanelProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const miniRef = useRef<HTMLCanvasElement | null>(null)
  const envRef = useRef(new SwarmEnv(400, 7))
  const policyRef = useRef<Policy | null>(null)
  const resetNextRef = useRef(false)
  const reviveNextRef = useRef(false)
  const [provider, setProvider] = useState('loading')
  const [alive, setAlive] = useState(N_AGENTS)
  const [coverage, setCoverage] = useState(0)
  const [status, setStatus] = useState(NOMINAL)
  const [lost, setLost] = useState(false)

  // setStatus lives in React; the render loop signals via this ref.
  const statusRef = useRef<{ text: string; lost: boolean } | null>(null)

  // Camera control state
  const [cameraMode, setCameraMode] = useState<'orbit' | 'drone-orbit' | 'fpv' | 'fps'>('orbit')
  const [selectedDroneIndex, setSelectedDroneIndex] = useState<number>(0)
  const [isPointerLocked, setIsPointerLocked] = useState(false)
  const [droneStates, setDroneStates] = useState<{ id: number; alive: boolean; x: number; y: number; vx: number; vy: number }[]>([])

  const cameraModeRef = useRef(cameraMode)
  const selectedDroneIndexRef = useRef(selectedDroneIndex)
  const keysPressed = useRef<{ [key: string]: boolean }>({})

  useEffect(() => {
    cameraModeRef.current = cameraMode
  }, [cameraMode])

  useEffect(() => {
    selectedDroneIndexRef.current = selectedDroneIndex
  }, [selectedDroneIndex])

  // Key handlers for FPS movement
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      if (['w', 'a', 's', 'd', ' ', 'shift', 'q', 'e'].includes(key)) {
        keysPressed.current[key] = true
      }
    }
    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      if (['w', 'a', 's', 'd', ' ', 'shift', 'q', 'e'].includes(key)) {
        keysPressed.current[key] = false
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [])

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
    const onContextMenu = (ev: MouseEvent) => ev.preventDefault()
    renderer.domElement.addEventListener('contextmenu', onContextMenu)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.07
    controls.target.set(0, 1.1, 0)
    controls.minDistance = 5
    controls.maxDistance = 30
    controls.maxPolarAngle = Math.PI * 0.48
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.PAN,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.ROTATE,
    }
    controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    }

    // FPS mouse drag & pointer lock variables
    let isDragging = false
    let previousMousePosition = { x: 0, y: 0 }

    const onMouseDown = (e: MouseEvent) => {
      if (cameraModeRef.current !== 'fps') return
      isDragging = true
      previousMousePosition = { x: e.clientX, y: e.clientY }
      
      const canvas = renderer.domElement
      if (canvas.requestPointerLock) {
        canvas.requestPointerLock()
      }
    }

    const onMouseMove = (e: MouseEvent) => {
      if (cameraModeRef.current !== 'fps') return
      
      let dx = 0
      let dy = 0
      
      if (document.pointerLockElement === renderer.domElement) {
        dx = e.movementX
        dy = e.movementY
      } else if (isDragging) {
        dx = e.clientX - previousMousePosition.x
        dy = e.clientY - previousMousePosition.y
        previousMousePosition = { x: e.clientX, y: e.clientY }
      } else {
        return
      }
      
      const sensitivity = 0.0025
      const euler = new THREE.Euler(0, 0, 0, 'YXZ')
      euler.setFromQuaternion(camera.quaternion)
      
      euler.y -= dx * sensitivity
      euler.x -= dy * sensitivity
      
      // Clamp pitch to avoid turning upside down (85 degrees)
      euler.x = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, euler.x))
      
      camera.quaternion.setFromEuler(euler)
    }

    const onMouseUp = () => {
      isDragging = false
    }

    const handlePointerLockChange = () => {
      const canvas = renderer.domElement
      setIsPointerLocked(document.pointerLockElement === canvas)
    }

    renderer.domElement.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    document.addEventListener('pointerlockchange', handlePointerLockChange)

    scene.add(new THREE.HemisphereLight(0xc9f0ff, 0x111911, 1.45))
    const key = new THREE.DirectionalLight(0xffffff, 1.4)
    key.position.set(4, 10, 6)
    scene.add(key)

    const grid = new THREE.GridHelper(WORLD_HALF * 2, GRID, 0x365547, 0x16231f)
    scene.add(grid)

    const boundary = new THREE.Group()
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: 0x89f4c7,
      transparent: true,
      opacity: 0.72,
    })
    const lowerFrame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(WORLD_HALF * 2, 0.03, WORLD_HALF * 2)),
      edgeMaterial,
    )
    lowerFrame.position.y = 0.08
    const upperFrame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(WORLD_HALF * 2, 3.2, WORLD_HALF * 2)),
      new THREE.LineBasicMaterial({ color: 0x396b5b, transparent: true, opacity: 0.48 }),
    )
    upperFrame.position.y = 1.6
    boundary.add(lowerFrame, upperFrame)
    scene.add(boundary)

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
      new THREE.BufferGeometry().setFromPoints(
        fakeTrajectory().map((p) => scenePoint(p.x, p.y, p.z)),
      ),
      new THREE.LineBasicMaterial({ color: 0x55b8ff, transparent: true, opacity: 0.34 }),
    )
    scene.add(trajectoryLine)

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

    let frameTimer = 0
    let disposed = false
    let stepping = false
    void loadTrajectory(trajectoryUrl).then((loaded) => {
      const vertices = loaded.map((p) => scenePoint(p.x, p.y, p.z))
      trajectoryLine.geometry.dispose()
      trajectoryLine.geometry = new THREE.BufferGeometry().setFromPoints(vertices)
    })
    void loadSplatPoints(splatPointsUrl).then((points) => {
      if (!disposed) scene.add(points)
    })

    let raf = 0
    let last = performance.now()
    let lastUIUpdate = 0
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
      if (frameTimer >= FRAME_MS && !stepping) {
        frameTimer = 0
        stepping = true
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

      // Update drone states for UI telemetry
      if (now - lastUIUpdate > 80) {
        lastUIUpdate = now
        const states = Array.from({ length: env.n }, (_, i) => ({
          id: i,
          alive: env.alive[i],
          x: env.pos[i * 2],
          y: env.pos[i * 2 + 1],
          vx: env.vel[i * 2] * MAX_SPEED,
          vy: env.vel[i * 2 + 1] * MAX_SPEED,
        }))
        setDroneStates(states)
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

      // status line signaled from operator actions in the render loop.
      if (statusRef.current) {
        setStatus(statusRef.current.text)
        setLost(statusRef.current.lost)
        statusRef.current = null
      }

      // Camera control modes update
      const currentMode = cameraModeRef.current
      const selectedDroneIdx = selectedDroneIndexRef.current

      if (currentMode === 'orbit') {
        controls.enabled = true
        controls.target.lerp(new THREE.Vector3(0, 1.1, 0), 0.1)
        controls.update()
      } else if (currentMode === 'drone-orbit') {
        controls.enabled = true
        const targetDrone = drones[selectedDroneIdx]
        if (targetDrone) {
          controls.target.lerp(targetDrone.group.position, 0.15)
        }
        controls.update()
      } else if (currentMode === 'fpv') {
        controls.enabled = false
        const targetDrone = drones[selectedDroneIdx]
        if (targetDrone) {
          // FPV position: slightly above and forward in the drone's local coordinate space
          const localOffset = new THREE.Vector3(0, 0.12, -0.32)
          const worldOffset = localOffset.clone().applyQuaternion(targetDrone.group.quaternion)
          const camPos = targetDrone.group.position.clone().add(worldOffset)
          
          camera.position.lerp(camPos, 0.25)
          
          // Look direction: local forward (0, 0, -1)
          const localForward = new THREE.Vector3(0, 0, -1)
          const worldForward = localForward.clone().applyQuaternion(targetDrone.group.quaternion)
          const lookTarget = camera.position.clone().add(worldForward)
          
          const tempMatrix = new THREE.Matrix4()
          tempMatrix.lookAt(camera.position, lookTarget, new THREE.Vector3(0, 1, 0))
          const targetRotation = new THREE.Quaternion().setFromRotationMatrix(tempMatrix)
          camera.quaternion.slerp(targetRotation, 0.2)
        }
      } else if (currentMode === 'fps') {
        controls.enabled = false
        // WASD key movement
        const speed = 7.0 * dt
        const moveDir = new THREE.Vector3()
        
        if (keysPressed.current['w']) {
          const f = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion)
          moveDir.add(f)
        }
        if (keysPressed.current['s']) {
          const b = new THREE.Vector3(0, 0, 1).applyQuaternion(camera.quaternion)
          moveDir.add(b)
        }
        if (keysPressed.current['a']) {
          const l = new THREE.Vector3(-1, 0, 0).applyQuaternion(camera.quaternion)
          moveDir.add(l)
        }
        if (keysPressed.current['d']) {
          const r = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion)
          moveDir.add(r)
        }
        if (keysPressed.current[' '] || keysPressed.current['e']) {
          moveDir.y += 1.0
        }
        if (keysPressed.current['shift'] || keysPressed.current['q']) {
          moveDir.y -= 1.0
        }
        
        if (moveDir.lengthSq() > 0) {
          moveDir.normalize().multiplyScalar(speed)
          camera.position.add(moveDir)
          
          // boundary checks
          camera.position.x = Math.max(-WORLD_HALF * 1.5, Math.min(WORLD_HALF * 1.5, camera.position.x))
          camera.position.y = Math.max(0.1, Math.min(15.0, camera.position.y))
          camera.position.z = Math.max(-WORLD_HALF * 1.5, Math.min(WORLD_HALF * 1.5, camera.position.z))
        }
      }

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
      renderer.domElement.removeEventListener('contextmenu', onContextMenu)
      
      // Clean up pointer lock and mouse listeners
      renderer.domElement.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      document.removeEventListener('pointerlockchange', handlePointerLockChange)

      controls.dispose()
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

  const selectedDroneState = droneStates[selectedDroneIndex]

  const handleKillDrone = (idx: number) => {
    const env = envRef.current
    if (env) {
      env.kill(idx)
      setAlive(env.nAlive())
      statusRef.current = { text: `UAV-0${idx + 1} propulsion system disabled · lost link`, lost: true }
    }
  }

  const handleReviveDrone = (idx: number) => {
    const env = envRef.current
    if (env) {
      env.revive(idx)
      setAlive(env.nAlive())
      statusRef.current = { text: `UAV-0${idx + 1} booted · link re-established`, lost: false }
    }
  }

  return (
    <section className="composite-scene" style={{ position: 'relative' }}>
      <div ref={mountRef} className="composite-canvas" />

      {/* TACTICAL CAMERA CONSOLE */}
      <div className="tactical-console">
        <div className="console-header">
          <span>◛ CAMERA SYSTEM</span>
          <span style={{ fontSize: 9, opacity: 0.6 }}>CTRL_DECK</span>
        </div>

        <div className="console-section-title">Select View Mode</div>
        <div className="console-btn-grid">
          <button
            type="button"
            className={`console-btn ${cameraMode === 'orbit' ? 'active' : ''}`}
            onClick={() => setCameraMode('orbit')}
          >
            GLOBAL ORBIT
          </button>
          <button
            type="button"
            className={`console-btn ${cameraMode === 'fps' ? 'active' : ''}`}
            onClick={() => setCameraMode('fps')}
          >
            FREE FLY (FPS)
          </button>
          <button
            type="button"
            className={`console-btn ${cameraMode === 'drone-orbit' ? 'active' : ''}`}
            onClick={() => setCameraMode('drone-orbit')}
          >
            DRONE TRACK
          </button>
          <button
            type="button"
            className={`console-btn ${cameraMode === 'fpv' ? 'active' : ''}`}
            onClick={() => setCameraMode('fpv')}
          >
            DRONE FPV
          </button>
        </div>

        {(cameraMode === 'drone-orbit' || cameraMode === 'fpv') && (
          <>
            <div className="console-section-title">Select UAV Platform</div>
            <div className="uav-grid">
              {Array.from({ length: N_AGENTS }).map((_, idx) => {
                const droneInfo = droneStates[idx]
                const isAlive = droneInfo ? droneInfo.alive : true
                return (
                  <button
                    key={idx}
                    type="button"
                    className={`uav-btn ${selectedDroneIndex === idx ? 'active' : ''}`}
                    onClick={() => setSelectedDroneIndex(idx)}
                  >
                    0{idx + 1}
                    <span className={`status-dot ${isAlive ? 'alive' : 'dead'}`} />
                  </button>
                )
              })}
            </div>

            {selectedDroneState && (
              <div className="telemetry-box">
                <div style={{ fontWeight: 'bold', color: '#89f4c7', marginBottom: 4, fontSize: 11 }}>
                  UAV-0{selectedDroneIndex + 1} TELEMETRY
                </div>
                <div className={`telemetry-row ${!selectedDroneState.alive ? 'warning' : ''}`}>
                  <span>STATUS:</span>
                  <span>{selectedDroneState.alive ? 'ONLINE' : 'LINK LOST'}</span>
                </div>
                <div className="telemetry-row">
                  <span>POSITION X:</span>
                  <span>{selectedDroneState.x.toFixed(2)}m</span>
                </div>
                <div className="telemetry-row">
                  <span>POSITION Y:</span>
                  <span>{selectedDroneState.y.toFixed(2)}m</span>
                </div>
                <div className="telemetry-row">
                  <span>ALTITUDE:</span>
                  <span>{selectedDroneState.alive ? '2.00m' : '0.00m'}</span>
                </div>
                <div className="telemetry-row">
                  <span>SPEED:</span>
                  <span>{Math.sqrt(selectedDroneState.vx ** 2 + selectedDroneState.vy ** 2).toFixed(2)} m/s</span>
                </div>

                <div className="telemetry-actions">
                  {selectedDroneState.alive ? (
                    <button
                      type="button"
                      className="btn-kill"
                      onClick={() => handleKillDrone(selectedDroneIndex)}
                    >
                      KILL UAV
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="btn-revive"
                      onClick={() => handleReviveDrone(selectedDroneIndex)}
                    >
                      REVIVE UAV
                    </button>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* FPV COCKPIT HUD OVERLAY */}
      {cameraMode === 'fpv' && (
        <div className="fpv-hud-overlay">
          <div className="fpv-hud-glass" />
          
          {selectedDroneState && !selectedDroneState.alive ? (
            <div className="fpv-hud-fault">
              <div className="fault-title">▸ UAV SIGNAL LOST / FAULT ◂</div>
              <div className="fault-desc">
                Propulsion hardware disabled or link jammed. Coordinates frozen. Check operator command panel to re-establish connection.
              </div>
            </div>
          ) : (
            <>
              <div className="fpv-hud-header">
                <div>FEED: UAV-0{selectedDroneIndex + 1} // FPV COCKPIT</div>
                <div className="rec-indicator">
                  <span className="rec-dot" />
                  <span>REC</span>
                </div>
              </div>

              {/* Center Crosshair */}
              <div className="fpv-hud-center">
                <div className="hud-bracket hud-bracket-tl" />
                <div className="hud-bracket hud-bracket-tr" />
                <div className="hud-bracket hud-bracket-bl" />
                <div className="hud-bracket hud-bracket-br" />
                <div className="hud-crosshair-center" />
              </div>

              {/* Altitude Tape (Left) */}
              <div className="fpv-hud-tape fpv-hud-tape-left">
                <div style={{ fontSize: 8, opacity: 0.6, letterSpacing: 0.5 }}>ALT (m)</div>
                <div className="tape-value">2.0</div>
                <div className="tape-ticks">
                  <span>- 3.0</span>
                  <span>- 2.5</span>
                  <span style={{ fontWeight: 'bold' }}>▸ 2.0</span>
                  <span>- 1.5</span>
                  <span>- 1.0</span>
                </div>
              </div>

              {/* Speed Tape (Right) */}
              <div className="fpv-hud-tape fpv-hud-tape-right">
                <div style={{ fontSize: 8, opacity: 0.6, letterSpacing: 0.5 }}>SPD (m/s)</div>
                <div className="tape-value">
                  {selectedDroneState ? Math.sqrt(selectedDroneState.vx ** 2 + selectedDroneState.vy ** 2).toFixed(1) : '0.0'}
                </div>
                <div className="tape-ticks">
                  <span>- 6.0</span>
                  <span>- 4.0</span>
                  <span>- 2.0</span>
                  <span style={{ fontWeight: 'bold' }}>▸ ACT</span>
                  <span>- 0.0</span>
                </div>
              </div>

              <div className="fpv-hud-footer">
                <div>GIMBAL PITCH: 0.0°</div>
                <div>YAW: {selectedDroneState ? (Math.atan2(selectedDroneState.vy, selectedDroneState.vx) * (180 / Math.PI)).toFixed(0) : '0'}°</div>
                <div>AUTONOMY STATUS: MAPPO_ACTIVE</div>
              </div>
            </>
          )}
        </div>
      )}

      {/* FPS CONTROLS HUD */}
      {cameraMode === 'fps' && (
        <>
          {!isPointerLocked && (
            <div
              className="fps-pointer-prompt"
              onClick={() => {
                const canvas = mountRef.current?.querySelector('canvas')
                if (canvas && canvas.requestPointerLock) {
                  canvas.requestPointerLock()
                }
              }}
            >
              [ CLICK TO ENGAGE FLIGHT CONTROLS ]
            </div>
          )}

          <div className="fps-instructions">
            <div className="fps-title">UAV SIM FLIGHT DECK</div>
            <div className="fps-keys">
              <div className="fps-key-row">
                <span>[W][A][S][D]</span>
                <span>MANEUVER</span>
              </div>
              <div className="fps-key-row">
                <span>[SPACE]</span>
                <span>ASCEND</span>
              </div>
              <div className="fps-key-row">
                <span>[L-SHIFT]</span>
                <span>DESCEND</span>
              </div>
              <div className="fps-key-row">
                <span>[MOUSE]</span>
                <span>LOOK AROUND</span>
              </div>
              <div className="fps-key-row">
                <span>[ESC]</span>
                <span>RELEASE MOUSE</span>
              </div>
            </div>
          </div>
        </>
      )}

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
          zIndex: 10,
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
          zIndex: 10,
        }}
      >
        {lost ? '▸ ' : '▸ '}
        {status}
      </div>

      <div className="mission-strip">
        <div>
          <strong>{missionName}</strong>
          <span>{missionBrief}</span>
        </div>
        <div>
          <span>GPS: DENIED</span>
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
        <button type="button" onClick={() => { reviveNextRef.current = true }}>
          Revive All
        </button>
        <button type="button" onClick={() => { resetNextRef.current = true }}>
          Reset
        </button>
      </div>
    </section>
  )
}
