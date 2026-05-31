import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { ALTITUDE, GRID, SwarmEnv, WORLD_HALF, MAX_SPEED } from '../swarm/sim'
import { makeDrone, updateDrone, Trail, type Drone } from '../swarm/drone'
import { drawMinimap, type MiniAgent } from '../swarm/minimap'
import { getScenarioDefaults } from '../gym/battlefieldParams'

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
  envId: string
  trajectoryUrl?: string
  missionName?: string
  swarmStreamUrl?: string
  cameraStreamUrl?: string
  policyEnabled?: boolean
  /**
   * Mock 3DGS point cloud by default. Swap-in seam for the REAL Gaussian splat
   * reconstructed from drone footage: point this at a JSON of {x,y,z,r,g,b}
   * (or replace `loadSplatPoints` with a true splat loader) and nothing else in
   * this scene — swarm, trajectory, coverage, controls — needs to change.
   */
  splatPointsUrl?: string
}

interface SwarmStreamAgent {
  id: number
  x: number
  y: number
  z: number
  yaw: number
  role?: string
  alive: boolean
}

interface SwarmStreamMessage {
  topic?: string
  t: number
  agents: SwarmStreamAgent[]
}

const FRAME_MS = 100
const CELL = (2 * WORLD_HALF) / GRID
const WORLD_Y_MIN = 0.35
const WORLD_Y_MAX = 11
const WORLD_XZ_LIMIT = WORLD_HALF
const TRAINED_COVERAGE = 0.9965
const RANDOM_COVERAGE = 0.4725

interface Waypoint {
  x: number
  y: number
}

// scene mapping: world (x,y,z) -> three (x = x, up = z, depth = -y)
function scenePoint(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, -y)
}

function worldToCell(x: number, y: number): [number, number] {
  let cx = Math.floor((x + WORLD_HALF) / CELL)
  let cy = Math.floor((y + WORLD_HALF) / CELL)
  if (cx < 0) cx = 0
  else if (cx > GRID - 1) cx = GRID - 1
  if (cy < 0) cy = 0
  else if (cy > GRID - 1) cy = GRID - 1
  return [cx, cy]
}

function markCoveredCells(covered: Uint8Array, agents: SwarmStreamAgent[]) {
  for (const agent of agents) {
    if (!agent.alive) continue
    const [cx, cy] = worldToCell(agent.x, agent.y)
    covered[cx * GRID + cy] = 1
  }
}

function clampWorldVector(point: THREE.Vector3, minY = WORLD_Y_MIN, maxY = WORLD_Y_MAX) {
  point.x = Math.max(-WORLD_XZ_LIMIT, Math.min(WORLD_XZ_LIMIT, point.x))
  point.y = Math.max(minY, Math.min(maxY, point.y))
  point.z = Math.max(-WORLD_XZ_LIMIT, Math.min(WORLD_XZ_LIMIT, point.z))
  return point
}

function wrapIndex(index: number, length: number) {
  return ((index % length) + length) % length
}

function routePoint(route: Waypoint[], phase: number): Waypoint {
  if (route.length === 0) return { x: 0, y: 0 }
  if (route.length === 1) return route[0]

  const raw = Math.floor(phase)
  const i = wrapIndex(raw, route.length)
  const a = route[i]
  const b = route[(i + 1) % route.length]
  const t = phase - raw
  const eased = t * t * (3 - 2 * t)
  return {
    x: a.x + (b.x - a.x) * eased,
    y: a.y + (b.y - a.y) * eased,
  }
}

function hashString(value: string) {
  let hash = 2166136261
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

function seededRandom(seed: number) {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function missionRoutes(envId: string): Waypoint[][] {
  if (envId === 'drone-vs-drone') {
    return [
      [{ x: -8, y: -4 }, { x: -4, y: -2 }, { x: -1.2, y: 0 }, { x: -4, y: 2 }, { x: -8, y: 4 }],
      [{ x: -8, y: 0 }, { x: -4, y: -1.6 }, { x: -0.8, y: 0.6 }, { x: -4, y: 1.7 }],
      [{ x: -7, y: 4 }, { x: -3, y: 3 }, { x: -1.4, y: 0.7 }, { x: -5, y: -2.8 }],
      [{ x: 8, y: -4 }, { x: 4, y: -2 }, { x: 1.4, y: 0 }, { x: 4, y: 2 }, { x: 8, y: 4 }],
      [{ x: 8, y: 0 }, { x: 4, y: -1.4 }, { x: 1, y: -0.5 }, { x: 4, y: 1.8 }],
      [{ x: 7, y: 4 }, { x: 3, y: 3 }, { x: 1.2, y: 0.6 }, { x: 5, y: -2.8 }],
    ]
  }

  if (envId === 'moving-target-track') {
    return [
      [{ x: -8, y: -2 }, { x: -3, y: -4 }, { x: 2, y: -3 }, { x: 7, y: -1 }, { x: 3, y: 2 }, { x: -4, y: 2 }],
      [{ x: -7, y: 3 }, { x: -2, y: 5 }, { x: 4, y: 4 }, { x: 8, y: 2 }, { x: 2, y: -1 }, { x: -5, y: -1 }],
      [{ x: -5, y: 0 }, { x: 0, y: -2 }, { x: 5, y: -1 }, { x: 6, y: 3 }, { x: 0, y: 4 }],
      [{ x: -6, y: -4 }, { x: -1, y: -5 }, { x: 5, y: -4 }, { x: 8, y: 0 }, { x: 1, y: 1 }],
    ]
  }

  if (envId === 'defend-asset') {
    return [
      [{ x: -4, y: 0 }, { x: -2, y: 3.5 }, { x: 2, y: 3.5 }, { x: 4, y: 0 }, { x: 2, y: -3.5 }, { x: -2, y: -3.5 }],
      [{ x: 0, y: 4.5 }, { x: 3.8, y: 1.5 }, { x: 2.3, y: -3.8 }, { x: -2.8, y: -3.6 }, { x: -4, y: 1.2 }],
      [{ x: 4.5, y: 0 }, { x: 1.5, y: -3.8 }, { x: -3.8, y: -2.2 }, { x: -3.5, y: 2.6 }, { x: 1.4, y: 4 }],
      [{ x: 0, y: -4.5 }, { x: -3.8, y: -1.5 }, { x: -2.3, y: 3.8 }, { x: 2.8, y: 3.6 }, { x: 4, y: -1.2 }],
      [{ x: -5.5, y: -5.5 }, { x: -4, y: 0 }, { x: -5.5, y: 5.5 }, { x: 0, y: 4 }, { x: 5.5, y: 5.5 }, { x: 4, y: 0 }, { x: 5.5, y: -5.5 }, { x: 0, y: -4 }],
    ]
  }

  if (envId === 'swarm-vs-swarm-race') {
    return [
      [{ x: -8, y: -7 }, { x: -5, y: -3 }, { x: -8, y: 1 }, { x: -5, y: 6 }, { x: -1, y: 4 }, { x: -2, y: -5 }],
      [{ x: -5, y: -8 }, { x: -2, y: -4 }, { x: -5, y: 1 }, { x: -1, y: 7 }, { x: 2, y: 3 }, { x: 1, y: -6 }],
      [{ x: -2, y: -7 }, { x: 1, y: -3 }, { x: -1, y: 2 }, { x: 2, y: 7 }, { x: 5, y: 2 }, { x: 4, y: -5 }],
      [{ x: 8, y: 7 }, { x: 5, y: 3 }, { x: 8, y: -1 }, { x: 5, y: -6 }, { x: 1, y: -4 }, { x: 2, y: 5 }],
      [{ x: 5, y: 8 }, { x: 2, y: 4 }, { x: 5, y: -1 }, { x: 1, y: -7 }, { x: -2, y: -3 }, { x: -1, y: 6 }],
      [{ x: 2, y: 7 }, { x: -1, y: 3 }, { x: 1, y: -2 }, { x: -2, y: -7 }, { x: -5, y: -2 }, { x: -4, y: 5 }],
    ]
  }

  return [
    [{ x: -8, y: -7 }, { x: -3, y: -7 }, { x: 2, y: -6 }, { x: 8, y: -5 }, { x: 8, y: -1 }, { x: 2, y: -1 }, { x: -5, y: -2 }],
    [{ x: -8, y: 2 }, { x: -3, y: 3 }, { x: 2, y: 2 }, { x: 8, y: 1 }, { x: 8, y: 6 }, { x: 2, y: 7 }, { x: -6, y: 6 }],
    [{ x: -7, y: -3 }, { x: -3, y: -2 }, { x: 0, y: 0 }, { x: 4, y: 1 }, { x: 8, y: 4 }],
    [{ x: 7, y: 7 }, { x: 4, y: 4 }, { x: 1, y: 1 }, { x: -2, y: 0 }, { x: -6, y: 1 }],
    [{ x: -6, y: 7 }, { x: -1, y: 5 }, { x: 4, y: 6 }, { x: 7, y: 2 }, { x: 3, y: -2 }, { x: -2, y: -4 }],
  ]
}

function scriptedMissionActions(env: SwarmEnv, envId: string, now: number): Float32Array {
  const routes = missionRoutes(envId)
  const actions = new Float32Array(env.n * 2)
  const phaseBase = now / 5200
  const stepScale = MAX_SPEED * 0.1

  for (let i = 0; i < env.n; i++) {
    if (!env.alive[i]) continue
    const route = routes[i % routes.length]
    const target = routePoint(route, phaseBase + i * 0.18)
    const px = env.pos[i * 2]
    const py = env.pos[i * 2 + 1]
    let ax = (target.x - px) / stepScale
    let ay = (target.y - py) / stepScale

    for (let j = 0; j < env.n; j++) {
      if (i === j || !env.alive[j]) continue
      const dx = px - env.pos[j * 2]
      const dy = py - env.pos[j * 2 + 1]
      const d2 = dx * dx + dy * dy
      if (d2 > 0.0001 && d2 < 2.8) {
        const push = (2.8 - d2) / 2.8
        ax += (dx / Math.sqrt(d2)) * push * 0.85
        ay += (dy / Math.sqrt(d2)) * push * 0.85
      }
    }

    const edge = WORLD_HALF - 1.2
    if (px > edge) ax -= 1.1
    if (px < -edge) ax += 1.1
    if (py > edge) ay -= 1.1
    if (py < -edge) ay += 1.1

    const mag = Math.max(1, Math.hypot(ax, ay))
    actions[i * 2] = Math.max(-1, Math.min(1, ax / mag))
    actions[i * 2 + 1] = Math.max(-1, Math.min(1, ay / mag))
  }

  return actions
}

function getMissionSimParams(envId: string) {
  const params = getScenarioDefaults(envId)
  return {
    ...params,
    logistics: {
      ...params.logistics,
      attritionInjectRate: 0,
    },
  }
}

function seedMissionEnv(env: SwarmEnv, envId: string) {
  env.reset(hashString(envId))
  env.vel.fill(0)
  env.covered.fill(0)
  env.reviveAll()

  const routes = missionRoutes(envId)
  for (let i = 0; i < env.n; i++) {
    const start = routes[i % routes.length]?.[0] ?? { x: 0, y: 0 }
    env.pos[i * 2] = start.x
    env.pos[i * 2 + 1] = start.y
  }

  env.steps = 0
  env.step(new Float32Array(env.n * 2))
  env.steps = 0
  env.vel.fill(0)
}

function createMissionEnv(envId: string) {
  const env = new SwarmEnv(400, 7, getMissionSimParams(envId))
  seedMissionEnv(env, envId)
  return env
}

function makeOrbitPan(keys: Record<string, boolean>, dt: number) {
  const pan = new THREE.Vector3()
  const panSpeed = 8.5 * dt
  if (keys.arrowleft) pan.x -= panSpeed
  if (keys.arrowright) pan.x += panSpeed
  if (keys.arrowup) pan.z -= panSpeed
  if (keys.arrowdown) pan.z += panSpeed
  if (keys.r || keys.e) pan.y += panSpeed
  if (keys.f || keys.q) pan.y -= panSpeed
  return pan
}

function getControlLegend(cameraMode: 'drone-orbit' | 'fpv') {
  if (cameraMode === 'drone-orbit') {
    return [
      'LMB rotate around tracked drone',
      'RMB pan orbit anchor',
      'Wheel zoom',
      'Arrows move X/Z',
      'R/E up · F/Q down',
    ]
  }
  if (cameraMode === 'fpv') {
    return [
      'Drone camera locked to selected unit',
      'Switch to Drone Track for pan or zoom',
      'Use UAV selector to change feed',
    ]
  }
  return []
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
function makeMockSplat(envId: string) {
  const rand = seededRandom(hashString(`splat:${envId}`))
  const count = 2400
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    const r = Math.sqrt(rand()) * WORLD_HALF * 0.82
    const theta = rand() * Math.PI * 2
    const x = Math.cos(theta) * r
    const y = Math.sin(theta) * r
    const ridge = Math.sin(x * 0.8) * Math.cos(y * 0.55) * 0.45
    const z = -0.08 + ridge + rand() * 0.16
    const p = scenePoint(x, y, z)
    positions.set([p.x, p.y, p.z], i * 3)
    const soil = 0.3 + rand() * 0.18
    colors.set([soil, 0.44 + rand() * 0.22, 0.24 + rand() * 0.08], i * 3)
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

async function loadSplatPoints(url: string | undefined, envId: string): Promise<THREE.Points> {
  if (!url) return makeMockSplat(envId)
  try {
    const res = await fetch(url)
    if (!res.ok) throw new Error(`${res.status}`)
    const data = await res.json()
    const points = Array.isArray(data) ? data : data.points
    if (!Array.isArray(points)) return makeMockSplat(envId)
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
    return makeMockSplat(envId)
  }
}


const NOMINAL = 'mission controller active · task behavior executing'
const POLICY_REQUIRED = 'train and export policy before mission launch'

export default function CompositeScenePanel({
  envId,
  trajectoryUrl,
  splatPointsUrl,
  missionName = 'Land Coverage Survey',
  swarmStreamUrl = 'ws://localhost:8765',
  cameraStreamUrl = 'ws://localhost:8000',
  policyEnabled = false,
}: CompositeScenePanelProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const miniRef = useRef<HTMLCanvasElement | null>(null)
  const envRef = useRef(createMissionEnv(envId))
  const missionActiveRef = useRef(false)
  const resetNextRef = useRef(false)
  const reviveNextRef = useRef(false)
  const [provider, setProvider] = useState('loading')
  const [alive, setAlive] = useState(0)
  const [coverage, setCoverage] = useState(0)
  const [meanAction, setMeanAction] = useState(0)
  const [status, setStatus] = useState(NOMINAL)
  const [lost, setLost] = useState(false)
  const [swarmLive, setSwarmLive] = useState(false)
  const [fpvFrame, setFpvFrame] = useState<string | null>(null)

  // setStatus lives in React; the render loop signals via this ref.
  const statusRef = useRef<{ text: string; lost: boolean } | null>(null)
  const liveSwarmRef = useRef<{
    connected: boolean
    lastUpdate: number
    agents: SwarmStreamAgent[]
  }>({ connected: false, lastUpdate: 0, agents: [] })
  const liveCoverageRef = useRef(new Uint8Array(GRID * GRID))

  // Camera control state
  const [cameraMode, setCameraMode] = useState<'drone-orbit' | 'fpv'>('drone-orbit')
  const [selectedDroneIndex, setSelectedDroneIndex] = useState<number>(0)
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
  const trackKey = (key: string, down: boolean) => {
    keysPressed.current[key] = down
  }

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      if (['w', 'a', 's', 'd', ' ', 'shift', 'q', 'e', 'r', 'f'].includes(key)) {
        e.preventDefault()
        trackKey(key, true)
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        trackKey('arrowup', true)
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        trackKey('arrowdown', true)
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        trackKey('arrowleft', true)
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        trackKey('arrowright', true)
      }
    }
    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      if (['w', 'a', 's', 'd', ' ', 'shift', 'q', 'e', 'r', 'f'].includes(key)) {
        e.preventDefault()
        trackKey(key, false)
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        trackKey('arrowup', false)
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        trackKey('arrowdown', false)
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        trackKey('arrowleft', false)
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        trackKey('arrowright', false)
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
    if (!policyEnabled) {
      missionActiveRef.current = false
      setProvider('required')
      setAlive(0)
      setCoverage(0)
      setMeanAction(0)
      setStatus(POLICY_REQUIRED)
      setLost(true)
      return
    }

    missionActiveRef.current = true
    seedMissionEnv(envRef.current, envId)
    liveCoverageRef.current.fill(0)
    liveSwarmRef.current = { connected: false, lastUpdate: 0, agents: [] }
    setProvider('scripted')
    setAlive(envRef.current.nAlive())
    setCoverage(envRef.current.coverage())
    setMeanAction(0)
    setStatus(NOMINAL)
    setLost(false)
  }, [envId, policyEnabled])

  useEffect(() => {
    if (!policyEnabled) return

    let ws: WebSocket | null = null
    let retry: number | null = null
    let closed = false

    const connect = () => {
      ws = new WebSocket(swarmStreamUrl)
      ws.onopen = () => {
        liveCoverageRef.current.fill(0)
        setSwarmLive(true)
        setProvider('pybullet-live')
        setStatus('pybullet swarm stream live · bus-authoritative motion')
        setLost(false)
      }
      ws.onclose = () => {
        liveSwarmRef.current.connected = false
        setSwarmLive(false)
        if (!closed) {
          setProvider('scripted')
          setStatus('waiting for pybullet swarm stream · local fallback active')
          retry = window.setTimeout(connect, 1500)
        }
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data as string) as SwarmStreamMessage
          if (msg.topic && msg.topic !== 'swarm') return
          if (!Array.isArray(msg.agents)) return
          liveSwarmRef.current = {
            connected: true,
            lastUpdate: performance.now(),
            agents: msg.agents,
          }
          markCoveredCells(liveCoverageRef.current, msg.agents)
        } catch {
          // ignore malformed frames
        }
      }
    }

    connect()
    return () => {
      closed = true
      if (retry != null) window.clearTimeout(retry)
      ws?.close()
    }
  }, [policyEnabled, swarmStreamUrl, envId])

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry: number | null = null
    let closed = false

    const connect = () => {
      ws = new WebSocket(cameraStreamUrl)
      ws.onclose = () => {
        if (!closed) retry = window.setTimeout(connect, 2000)
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data as string) as { topic?: string; data?: string }
          if (msg.topic !== 'fpv_raw' && msg.topic !== 'camera_frame') return
          if (typeof msg.data !== 'string') return
          setFpvFrame(`data:image/jpeg;base64,${msg.data}`)
        } catch {
          // ignore malformed frames
        }
      }
    }

    connect()
    return () => {
      closed = true
      if (retry != null) window.clearTimeout(retry)
      ws?.close()
    }
  }, [cameraStreamUrl])

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
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    }
    controls.screenSpacePanning = true
    controls.keyPanSpeed = 28
    controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    }
    controls.enablePan = true
    controls.enableZoom = true
    controls.zoomSpeed = 0.9
    controls.panSpeed = 0.95
    controls.rotateSpeed = 0.8

    let droneOrbitOffset = new THREE.Vector3(0, 0, 0)
    let previousMode: 'drone-orbit' | 'fpv' = cameraModeRef.current

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
    const refreshCoverage = (covered: Uint8Array) => {
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
    trajectoryLine.visible = false
    scene.add(trajectoryLine)

    // --- drones (quadrotor meshes + trails from drone.ts) ---
    const drones: Drone[] = Array.from({ length: env0.n }, () => makeDrone())
    drones.forEach((d) => {
      d.group.visible = false
      scene.add(d.group)
    })
    const actionArrows = drones.map(() => {
      const arrow = new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, ALTITUDE + 0.35, 0),
        1.8,
        0x7fd9ff,
        0.45,
        0.22,
      )
      arrow.visible = false
      scene.add(arrow)
      return arrow
    })
    const current: THREE.Vector3[] = drones.map((_, i) =>
      scenePoint(env0.pos[i * 2], env0.pos[i * 2 + 1], ALTITUDE),
    )
    const trails = drones.map(() => {
      const t = new Trail(0x4df09a)
      t.line.visible = false
      scene.add(t.line)
      return t
    })

    let frameTimer = 0
    let disposed = false
    let stepping = false
    let splatObject: THREE.Points | null = null
    void loadTrajectory(trajectoryUrl).then((loaded) => {
      const vertices = loaded.map((p) => scenePoint(p.x, p.y, p.z))
      trajectoryLine.geometry.dispose()
      trajectoryLine.geometry = new THREE.BufferGeometry().setFromPoints(vertices)
    })
    void loadSplatPoints(splatPointsUrl, envId).then((points) => {
      if (!disposed) {
        points.visible = missionActiveRef.current
        splatObject = points
        scene.add(points)
      }
    })

    let raf = 0
    let last = performance.now()
    const missionStartedAt = last
    let lastUIUpdate = 0
    const animate = (now: number) => {
      raf = requestAnimationFrame(animate)
      const delta = now - last
      last = now
      const dt = delta / 1000
      frameTimer += delta

      const env = envRef.current
      const missionLaunched = missionActiveRef.current
      const liveSwarmFresh =
        missionLaunched &&
        liveSwarmRef.current.connected &&
        now - liveSwarmRef.current.lastUpdate < 1500 &&
        liveSwarmRef.current.agents.length > 0
      const liveAgents = liveSwarmFresh ? liveSwarmRef.current.agents : []

      coverMesh.visible = missionLaunched
      trajectoryLine.visible = missionLaunched
      if (splatObject) splatObject.visible = missionLaunched
      drones.forEach((drone, i) => {
        drone.group.visible = missionLaunched
        trails[i].line.visible = missionLaunched
        if (!missionLaunched) actionArrows[i].visible = false
      })
      if (!missionLaunched && lastCoveredCount !== 0) {
        for (let idx = 0; idx < GRID * GRID; idx++) coverMesh.setMatrixAt(idx, HIDDEN)
        coverMesh.instanceMatrix.needsUpdate = true
        lastCoveredCount = 0
      }

      if (resetNextRef.current) {
        resetNextRef.current = false
        seedMissionEnv(env, envId)
        trails.forEach((t) => t.clear())
        lastCoveredCount = -1
        liveCoverageRef.current.fill(0)
        for (let idx = 0; idx < GRID * GRID; idx++) coverMesh.setMatrixAt(idx, HIDDEN)
        coverMesh.instanceMatrix.needsUpdate = true
        statusRef.current = missionLaunched
          ? { text: NOMINAL, lost: false }
          : { text: POLICY_REQUIRED, lost: true }
      }
      if (reviveNextRef.current) {
        reviveNextRef.current = false
        env.reviveAll()
        statusRef.current = missionLaunched
          ? { text: NOMINAL, lost: false }
          : { text: POLICY_REQUIRED, lost: true }
      }
      if (frameTimer >= FRAME_MS) {
        frameTimer = 0
        if (!stepping && missionLaunched && !liveSwarmFresh) {
          stepping = true
          const actions = scriptedMissionActions(env, envId, now - missionStartedAt)
          env.step(actions)
          setAlive(env.nAlive())
          setCoverage(env.coverage())
          let total = 0
          for (let i = 0; i < env.n; i++) {
            total += Math.hypot(actions[i * 2], actions[i * 2 + 1])
          }
          setMeanAction(total / env.n)
          refreshCoverage(env.coveredCells())
          stepping = false
        } else if (missionLaunched && liveSwarmFresh) {
          refreshCoverage(liveCoverageRef.current)
          setAlive(liveAgents.filter((agent) => agent.alive).length)
          const liveCovered = liveCoverageRef.current.reduce((sum, value) => sum + value, 0)
          setCoverage(liveCovered / (GRID * GRID))
          setMeanAction(0.45)
        }
      }

      // drones: smooth toward target, then drone.ts handles spin + dead-state.
      // updateDrone(x, y, z) places at position.set(x, z, y), so we pass the
      // scene vector as (sx, sz, sy) to land at the intended (sx, sy, sz).
      if (missionLaunched) for (let i = 0; i < env.n; i++) {
        const liveAgent = liveAgents.find((agent) => agent.id === i)
        const px = liveAgent ? liveAgent.x : env.pos[i * 2]
        const py = liveAgent ? liveAgent.y : env.pos[i * 2 + 1]
        const pz = liveAgent ? liveAgent.z : ALTITUDE + Math.sin(now * 0.002 + i) * 0.18
        const target = scenePoint(px, py, pz)
        current[i].lerp(target, liveSwarmFresh ? 0.34 : 0.26)
        const yaw = liveAgent ? -liveAgent.yaw : -Math.atan2(env.vel[i * 2 + 1], env.vel[i * 2])
        const droneAlive = liveAgent ? liveAgent.alive : env.alive[i]
        updateDrone(drones[i], current[i].x, current[i].z, current[i].y, yaw, droneAlive, dt)
        if (droneAlive) trails[i].push(current[i].x, current[i].z, current[i].y)
        const vx = liveAgent ? Math.cos(liveAgent.yaw) : env.vel[i * 2]
        const vy = liveAgent ? Math.sin(liveAgent.yaw) : env.vel[i * 2 + 1]
        const speed = liveAgent ? 0.5 : Math.hypot(vx, vy)
        const arrow = actionArrows[i]
        arrow.visible = droneAlive && speed > 0.03
        arrow.position.set(current[i].x, current[i].y + 0.45, current[i].z)
        if (speed > 0.03) {
          arrow.setDirection(new THREE.Vector3(vx, 0, -vy).normalize())
          arrow.setLength(1.0 + speed * 1.4, 0.45, 0.22)
        }
      }

      // Update drone states for UI telemetry
      if (now - lastUIUpdate > 80) {
        lastUIUpdate = now
        const states = missionLaunched
          ? liveSwarmFresh
            ? liveAgents.map((agent) => ({
                id: agent.id,
                alive: agent.alive,
                x: agent.x,
                y: agent.y,
                vx: Math.cos(agent.yaw) * MAX_SPEED * 0.45,
                vy: Math.sin(agent.yaw) * MAX_SPEED * 0.45,
              }))
            : Array.from({ length: env.n }, (_, i) => ({
                id: i,
                alive: env.alive[i],
                x: env.pos[i * 2],
                y: env.pos[i * 2 + 1],
                vx: env.vel[i * 2] * MAX_SPEED,
                vy: env.vel[i * 2 + 1] * MAX_SPEED,
              }))
          : []
        setDroneStates(states)
      }

      // minimap (top-down coordination map)
      const mini = miniRef.current
      if (mini) {
        const ctx = mini.getContext('2d')
        if (ctx) {
          const agents: MiniAgent[] = []
          if (missionLaunched) {
            if (liveSwarmFresh) {
              for (const agent of liveAgents) {
                agents.push({ x: agent.x, y: agent.y, alive: agent.alive })
              }
            } else {
              for (let i = 0; i < env.n; i++) {
                agents.push({ x: env.pos[i * 2], y: env.pos[i * 2 + 1], alive: env.alive[i] })
              }
            }
          }
          drawMinimap(
            ctx,
            mini.width,
            WORLD_HALF,
            GRID,
            missionLaunched
              ? (liveSwarmFresh ? liveCoverageRef.current : env.coveredCells())
              : new Uint8Array(GRID * GRID),
            agents,
          )
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

      const applyKeyboardPan = () => {
        const pan = makeOrbitPan(keysPressed.current, dt)
        if (pan.lengthSq() > 0) {
          controls.target.add(pan)
          camera.position.add(pan)
          clampWorldVector(controls.target)
          clampWorldVector(camera.position, WORLD_Y_MIN, WORLD_Y_MAX)
        }
      }

      if (currentMode !== previousMode) {
        if (currentMode === 'drone-orbit') {
          const targetDrone = drones[selectedDroneIdx]
          droneOrbitOffset = targetDrone
            ? controls.target.clone().sub(targetDrone.group.position)
            : new THREE.Vector3(0, 0, 0)
        }
        previousMode = currentMode
      }

      if (currentMode === 'drone-orbit') {
        controls.enabled = true
        const targetDrone = drones[selectedDroneIdx]
        if (targetDrone) {
          const anchoredTarget = targetDrone.group.position.clone().add(droneOrbitOffset)
          controls.target.copy(anchoredTarget)
        }
        applyKeyboardPan()
        controls.update()
        clampWorldVector(controls.target)
        clampWorldVector(camera.position, WORLD_Y_MIN, WORLD_Y_MAX)
        if (targetDrone) {
          droneOrbitOffset.copy(controls.target).sub(targetDrone.group.position)
        }
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
       // Clean up resize listener
       window.removeEventListener('resize', onResize)
       renderer.domElement.removeEventListener('contextmenu', onContextMenu)

      controls.dispose()
      actionArrows.forEach((arrow) => {
        arrow.line.geometry.dispose()
        arrow.cone.geometry.dispose()
      })
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
  }, [envId, splatPointsUrl, trajectoryUrl])

  const selectedDroneState = droneStates[selectedDroneIndex]
  const controlLegend = getControlLegend(cameraMode)
  const localControlsDisabled = swarmLive

  const handleKillDrone = (idx: number) => {
    if (localControlsDisabled) return
    const env = envRef.current
    if (env) {
      env.kill(idx)
      setAlive(env.nAlive())
      statusRef.current = { text: `UAV-0${idx + 1} propulsion system disabled · lost link`, lost: true }
    }
  }

  const handleReviveDrone = (idx: number) => {
    if (localControlsDisabled) return
    const env = envRef.current
    if (env) {
      env.revive(idx)
      setAlive(env.nAlive())
      statusRef.current = { text: `UAV-0${idx + 1} booted · link re-established`, lost: false }
    }
  }

  return (
    <section className="composite-scene" aria-label={`${missionName} mission simulation`}>
      <div ref={mountRef} className="composite-canvas" />

      <div className="composite-hud" aria-label="Mission HUD">
        <div className={`composite-hud__status${lost ? ' is-lost' : ''}`} role="status">
          {status}
        </div>

        <aside className="composite-hud__left">
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
        <p className="console-hint">
          Drone track orbits around the selected drone; left drag rotates, right drag adjusts offset, and
          the wheel zooms. FPV locks the viewport to the selected drone's camera perspective.
        </p>

        <div className="console-section-title">Control Legend</div>
        <div className="control-legend" role="list" aria-label="Camera controls">
          {controlLegend.map((entry) => (
            <div key={entry} className="control-legend-row" role="listitem">
              {entry}
            </div>
          ))}
        </div>

        {(cameraMode === 'drone-orbit' || cameraMode === 'fpv') && (
          <>
            <div className="console-section-title">Select UAV Platform</div>
            <div className="uav-grid">
              {Array.from({ length: envRef.current.n }).map((_, idx) => {
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
                {localControlsDisabled && (
                  <div className="telemetry-row">
                    <span>CONTROL:</span>
                    <span>LIVE BUS</span>
                  </div>
                )}
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
                      disabled={localControlsDisabled}
                      title={localControlsDisabled ? 'Kill is not wired into the live runtime yet.' : undefined}
                    >
                      KILL UAV
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="btn-revive"
                      onClick={() => handleReviveDrone(selectedDroneIndex)}
                      disabled={localControlsDisabled}
                      title={localControlsDisabled ? 'Revive is not wired into the live runtime yet.' : undefined}
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
        </aside>

        <aside className="composite-hud__right">
          <div className="coordination-map">
            <div className="coordination-map__label">COORDINATION MAP</div>
            <canvas ref={miniRef} width={150} height={150} className="coordination-map__canvas" />
          </div>

          <div className="mission-metrics" aria-label="Mission telemetry">
            <span>BACKEND: {provider.toUpperCase()}</span>
            {provider !== 'loading' && provider !== 'required' && (
              <span>
                EVAL: {Math.round(TRAINED_COVERAGE * 100)}% / RANDOM {Math.round(RANDOM_COVERAGE * 100)}%
              </span>
            )}
            <span>CMD: {Math.round(meanAction * 100)}%</span>
            <span>ALIVE: {alive}</span>
            <span>COVERAGE: {Math.round(coverage * 100)}%</span>
          </div>
        </aside>

        <footer className="composite-hud__footer">
          <div className="scene-actions">
            <button
              type="button"
              onClick={() => { reviveNextRef.current = true }}
              disabled={localControlsDisabled}
              title={localControlsDisabled ? 'Live PyBullet stream is authoritative.' : undefined}
            >
              Revive All
            </button>
            <button
              type="button"
              onClick={() => { resetNextRef.current = true }}
              disabled={localControlsDisabled}
              title={localControlsDisabled ? 'Live PyBullet stream is authoritative.' : undefined}
            >
              Reset
            </button>
          </div>
        </footer>
      </div>

      {/* FPV COCKPIT HUD OVERLAY */}
      {cameraMode === 'fpv' && (
        <div className="fpv-hud-overlay">
          {fpvFrame && <img className="fpv-hud-frame" src={fpvFrame} alt="FPV stream" />}
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
                  <span>{fpvFrame ? 'LIVE' : 'SIM'}</span>
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
                <div>AUTONOMY STATUS: {swarmLive ? 'PYBULLET_LIVE' : 'SCRIPTED_FALLBACK'}</div>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}
