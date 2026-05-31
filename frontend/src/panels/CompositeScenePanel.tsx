import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { ALTITUDE, GRID, SwarmEnv, WORLD_HALF, MAX_SPEED } from '../swarm/sim'
import { loadPolicy, type Policy } from '../swarm/policy'
import { makeDrone, updateDrone, Trail, type Drone } from '../swarm/drone'
import { drawMinimap, type MiniAgent, type MiniMarker } from '../swarm/minimap'
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
  policyEnabled?: boolean
}

const FRAME_MS = 100
const CELL = (2 * WORLD_HALF) / GRID
const WORLD_Y_MIN = 0.35
const WORLD_Y_MAX = 11
const WORLD_XZ_LIMIT = WORLD_HALF

// scene mapping: world (x,y,z) -> three (x = x, up = z, depth = -y)
function scenePoint(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, -y)
}

function clampWorldVector(point: THREE.Vector3, minY = WORLD_Y_MIN, maxY = WORLD_Y_MAX) {
  point.x = Math.max(-WORLD_XZ_LIMIT, Math.min(WORLD_XZ_LIMIT, point.x))
  point.y = Math.max(minY, Math.min(maxY, point.y))
  point.z = Math.max(-WORLD_XZ_LIMIT, Math.min(WORLD_XZ_LIMIT, point.z))
  return point
}

function hashString(value: string) {
  let hash = 2166136261
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
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
  env.reviveAll()
}

function createMissionEnv(envId: string) {
  const env = new SwarmEnv(400, 7, getMissionSimParams(envId), envId)
  seedMissionEnv(env, envId)
  return env
}

function hasMissionTarget(scenarioId: string | null) {
  return scenarioId === 'moving-target-track' || scenarioId === 'search-and-interdict'
}

function getMissionMarkers(env: SwarmEnv): MiniMarker[] {
  const markers: MiniMarker[] = []

  if (hasMissionTarget(env.scenarioId)) {
    markers.push({
      kind: 'target',
      x: env.targetPos[0],
      y: env.targetPos[1],
      active: env.scenarioId !== 'search-and-interdict' || env.contactStep !== null,
    })
  }

  if (env.scenarioId === 'defend-asset') {
    markers.push({ kind: 'asset', x: env.assetPos[0], y: env.assetPos[1] })
  }

  for (let i = 0; i < env.hostileAlive.length; i++) {
    markers.push({
      kind: 'hostile',
      x: env.hostilePos[i * 2],
      y: env.hostilePos[i * 2 + 1],
      active: env.hostileAlive[i],
    })
  }

  for (let i = 0; i < env.rivalPos.length / 2; i++) {
    markers.push({
      kind: 'rival',
      x: env.rivalPos[i * 2],
      y: env.rivalPos[i * 2 + 1],
    })
  }

  return markers
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

const NOMINAL = 'pybullet environment active · trained policy context loaded'
const POLICY_REQUIRED = 'train and export policy before mission launch'

export default function CompositeScenePanel({
  envId,
  trajectoryUrl,
  missionName = 'Land Coverage Survey',
  policyEnabled = false,
}: CompositeScenePanelProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const miniRef = useRef<HTMLCanvasElement | null>(null)
  const envRef = useRef(createMissionEnv(envId))
  const policyRef = useRef<Policy | null>(null)
  const inferringRef = useRef(false)
  const missionActiveRef = useRef(false)
  const resetNextRef = useRef(false)
  const reviveNextRef = useRef(false)
  const [provider, setProvider] = useState('loading')
  const [alive, setAlive] = useState(0)
  const [coverage, setCoverage] = useState(0)
  const [meanAction, setMeanAction] = useState(0)
  const [status, setStatus] = useState(NOMINAL)
  const [lost, setLost] = useState(false)

  // setStatus lives in React; the render loop signals via this ref.
  const statusRef = useRef<{ text: string; lost: boolean } | null>(null)

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
    let canceled = false
    policyRef.current = null
    inferringRef.current = false
    envRef.current = createMissionEnv(envId)

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

    seedMissionEnv(envRef.current, envId)
    missionActiveRef.current = false
    setProvider('loading')
    setAlive(envRef.current.nAlive())
    setCoverage(envRef.current.coverage())
    setMeanAction(0)
    setStatus('loading trained policy controller')
    setLost(false)
    void loadPolicy(envId)
      .then((policy) => {
        if (canceled) return
        policyRef.current = policy
        missionActiveRef.current = true
        setProvider(policy.provider)
        setStatus(NOMINAL)
        setLost(false)
      })
      .catch((err) => {
        if (canceled) return
        policyRef.current = null
        missionActiveRef.current = false
        setProvider('failed')
        setStatus(`policy load failed · ${err instanceof Error ? err.message : 'unknown error'}`)
        setLost(true)
      })

    return () => {
      canceled = true
    }
  }, [envId, policyEnabled])

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

    const targetGroup = new THREE.Group()
    const targetRing = new THREE.Mesh(
      new THREE.RingGeometry(0.42, 0.7, 36),
      new THREE.MeshBasicMaterial({
        color: 0xffcf66,
        transparent: true,
        opacity: 0.9,
        side: THREE.DoubleSide,
      }),
    )
    targetRing.rotation.x = -Math.PI / 2
    const targetCore = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 20, 12),
      new THREE.MeshBasicMaterial({ color: 0xffcf66 }),
    )
    targetCore.position.y = 0.34
    targetGroup.add(targetRing, targetCore)
    targetGroup.visible = false
    scene.add(targetGroup)

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
    let stepping = false
    void loadTrajectory(trajectoryUrl).then((loaded) => {
      const vertices = loaded.map((p) => scenePoint(p.x, p.y, p.z))
      trajectoryLine.geometry.dispose()
      trajectoryLine.geometry = new THREE.BufferGeometry().setFromPoints(vertices)
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
      const missionLaunched = missionActiveRef.current

      coverMesh.visible = missionLaunched
      trajectoryLine.visible = missionLaunched
      targetGroup.visible = missionLaunched && hasMissionTarget(env.scenarioId)
      if (targetGroup.visible) {
        targetGroup.position.copy(scenePoint(env.targetPos[0], env.targetPos[1], 0.08))
        targetGroup.rotation.y = now * 0.0018
      }
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
      if (frameTimer >= FRAME_MS && !stepping && missionLaunched) {
        frameTimer = 0
        stepping = true
        const policy = policyRef.current
        if (policy && !inferringRef.current) {
          inferringRef.current = true
          const obs = env.observe()
          policy.act(obs, env.n)
            .then((actions) => {
              env.step(actions)
              setAlive(env.nAlive())
              setCoverage(env.coverage())
              let total = 0
              for (let i = 0; i < env.n; i++) {
                total += Math.hypot(actions[i * 2], actions[i * 2 + 1])
              }
              setMeanAction(total / env.n)
              refreshCoverage()
            })
            .catch((err) => {
              missionActiveRef.current = false
              statusRef.current = {
                text: `policy inference failed · ${err instanceof Error ? err.message : 'unknown error'}`,
                lost: true,
              }
              setProvider('failed')
            })
            .finally(() => {
              inferringRef.current = false
              stepping = false
            })
        } else {
          stepping = false
        }
      }

      // drones: smooth toward target, then drone.ts handles spin + dead-state.
      // updateDrone(x, y, z) places at position.set(x, z, y), so we pass the
      // scene vector as (sx, sz, sy) to land at the intended (sx, sy, sz).
      if (missionLaunched) for (let i = 0; i < env.n; i++) {
        const target = scenePoint(
          env.pos[i * 2],
          env.pos[i * 2 + 1],
          ALTITUDE + Math.sin(now * 0.002 + i) * 0.18,
        )
        current[i].lerp(target, 0.26)
        const yaw = -Math.atan2(env.vel[i * 2 + 1], env.vel[i * 2])
        updateDrone(drones[i], current[i].x, current[i].z, current[i].y, yaw, env.alive[i], dt)
        if (env.alive[i]) trails[i].push(current[i].x, current[i].z, current[i].y)
        const speed = Math.hypot(env.vel[i * 2], env.vel[i * 2 + 1])
        const arrow = actionArrows[i]
        arrow.visible = env.alive[i] && speed > 0.03
        arrow.position.set(current[i].x, current[i].y + 0.45, current[i].z)
        if (speed > 0.03) {
          arrow.setDirection(new THREE.Vector3(env.vel[i * 2], 0, -env.vel[i * 2 + 1]).normalize())
          arrow.setLength(1.0 + speed * 1.4, 0.45, 0.22)
        }
      }

      // Update drone states for UI telemetry
      if (now - lastUIUpdate > 80) {
        lastUIUpdate = now
        const states = missionLaunched
          ? Array.from({ length: env.n }, (_, i) => ({
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
            for (let i = 0; i < env.n; i++) {
              agents.push({ x: env.pos[i * 2], y: env.pos[i * 2 + 1], alive: env.alive[i] })
            }
          }
          drawMinimap(
            ctx,
            mini.width,
            WORLD_HALF,
            GRID,
            missionLaunched ? env.coveredCells() : new Uint8Array(GRID * GRID),
            agents,
            missionLaunched ? getMissionMarkers(env) : [],
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
  }, [envId, trajectoryUrl])

  const selectedDroneState = droneStates[selectedDroneIndex]
  const controlLegend = getControlLegend(cameraMode)

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
        </aside>

        <aside className="composite-hud__right">
          <div className="coordination-map">
            <div className="coordination-map__label">COORDINATION MAP</div>
            <canvas ref={miniRef} width={150} height={150} className="coordination-map__canvas" />
          </div>

          <div className="mission-metrics" aria-label="Mission telemetry">
            <span>ENV: PYBULLET SIM</span>
            <span>CTRL: {provider.toUpperCase()}</span>
            <span>CMD: {Math.round(meanAction * 100)}%</span>
            <span>ALIVE: {alive}</span>
            <span>COVERAGE: {Math.round(coverage * 100)}%</span>
          </div>
        </aside>

        <footer className="composite-hud__footer">
          <div className="scene-actions">
            <button type="button" onClick={() => { reviveNextRef.current = true }}>
              Revive All
            </button>
            <button type="button" onClick={() => { resetNextRef.current = true }}>
              Reset
            </button>
          </div>
        </footer>
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
    </section>
  )
}
