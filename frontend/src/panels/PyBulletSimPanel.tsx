import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { PYBULLET_DEMO_CONFIG } from '../gym/pybulletDemoConfig'
import { getScenarioById } from '../gym/scenarios'
import { PYBULLET_WS_URL, startPyBulletSim } from '../gym/trainApi'

interface AgentPose {
  id: number
  x: number
  y: number
  z: number
  yaw: number
  role?: string
  alive: boolean
}

interface SwarmMessage {
  topic: 'swarm'
  t: number
  env_id?: string
  policy?: string
  coverage?: number
  captures?: number
  contact?: boolean
  agents: AgentPose[]
}

interface PyBulletFrameMessage {
  topic: 'pybullet_frame'
  t: number
  env_id?: string
  width: number
  height: number
  encoding: 'jpeg' | 'rgba'
  camera_mode?: CameraMode
  selected_drone?: number
  data: string
}

interface PyBulletSimPanelProps {
  envId: string
  missionName: string
  wsUrl?: string
}

type ConnectionState = 'connecting' | 'online' | 'offline'
type CameraMode = 'observer' | 'chase' | 'fpv'


export default function PyBulletSimPanel({
  envId,
  missionName,
  wsUrl = PYBULLET_WS_URL,
}: PyBulletSimPanelProps) {
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [frameSrc, setFrameSrc] = useState<string | null>(null)
  const [hasCanvasFrame, setHasCanvasFrame] = useState(false)
  const [frameTime, setFrameTime] = useState(0)
  const [agents, setAgents] = useState<AgentPose[]>([])
  const [policy, setPolicy] = useState('trained')
  const [captures, setCaptures] = useState(0)
  const [contact, setContact] = useState(false)
  const [cameraMode, setCameraMode] = useState<CameraMode>('observer')
  const [selectedDrone, setSelectedDrone] = useState(0)
  const [switchingCamera, setSwitchingCamera] = useState(false)
  const reconnectTimer = useRef<number | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    let disposed = false
    let socket: WebSocket | null = null

    const connect = () => {
      if (disposed) return
      setConnection('connecting')
      socket = new WebSocket(wsUrl)

      socket.onopen = () => {
        if (!disposed) setConnection('online')
      }

      socket.onmessage = (event) => {
        if (disposed || typeof event.data !== 'string') return
        const message = JSON.parse(event.data) as SwarmMessage | PyBulletFrameMessage
        if (message.env_id && message.env_id !== envId) return

        if (message.topic === 'pybullet_frame') {
          if (message.encoding === 'rgba') {
            const canvas = canvasRef.current
            const ctx = canvas?.getContext('2d')
            if (canvas && ctx) {
              canvas.width = message.width
              canvas.height = message.height
              const raw = window.atob(message.data)
              const pixels = new Uint8ClampedArray(raw.length)
              for (let i = 0; i < raw.length; i += 1) {
                pixels[i] = raw.charCodeAt(i)
              }
              ctx.putImageData(new ImageData(pixels, message.width, message.height), 0, 0)
              setHasCanvasFrame(true)
              setFrameSrc(null)
            }
          } else {
            setFrameSrc(`data:image/jpeg;base64,${message.data}`)
            setHasCanvasFrame(false)
          }
          if (message.camera_mode) setCameraMode(message.camera_mode)
          if (typeof message.selected_drone === 'number') {
            setSelectedDrone(message.selected_drone)
          }
          setFrameTime(message.t)
          return
        }

        if (message.topic === 'swarm') {
          setAgents(message.agents)
          setPolicy(message.policy ?? 'trained')
          if (typeof message.captures === 'number') setCaptures(message.captures)
          if (typeof message.contact === 'boolean') setContact(message.contact)
        }
      }

      socket.onerror = () => {
        if (!disposed) setConnection('offline')
      }

      socket.onclose = () => {
        if (disposed) return
        setConnection('offline')
        reconnectTimer.current = window.setTimeout(connect, 900)
      }
    }

    connect()

    return () => {
      disposed = true
      if (reconnectTimer.current != null) {
        window.clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
      socket?.close()
    }
  }, [envId, wsUrl])

  const alive = useMemo(
    () => agents.filter((agent) => agent.alive).length,
    [agents],
  )

  const switchCamera = useCallback(async (nextMode: CameraMode, nextDrone = selectedDrone) => {
    setSwitchingCamera(true)
    setCameraMode(nextMode)
    setSelectedDrone(nextDrone)
    try {
      const result = await startPyBulletSim(envId, nextMode, nextDrone)
      if (!result.ok) setConnection('offline')
    } finally {
      setSwitchingCamera(false)
    }
  }, [envId, selectedDrone])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return
      const key = event.key.toLowerCase()
      if (key === 'c') {
        const modes: CameraMode[] = ['observer', 'chase', 'fpv']
        const nextMode = modes[(modes.indexOf(cameraMode) + 1) % modes.length]
        void switchCamera(nextMode)
      }
      if (key === 'b') void switchCamera('observer')
      if (key === 'h') void switchCamera('chase')
      if (key === 'f') void switchCamera('fpv')
      const digit = Number(key)
      if (Number.isInteger(digit) && digit >= 1 && digit <= 9) {
        void switchCamera(cameraMode === 'observer' ? 'chase' : cameraMode, digit - 1)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [cameraMode, selectedDrone, switchCamera])

  const scenario = getScenarioById(envId)
  const simConfig = PYBULLET_DEMO_CONFIG.simulation
  const selectedAgent = agents.find((agent) => agent.id === selectedDrone)
  const speed = selectedAgent
    ? Math.hypot(Math.cos(selectedAgent.yaw), Math.sin(selectedAgent.yaw)) * 4.2
    : 0
  const modeLabel = cameraMode === 'fpv' ? 'FPV' : cameraMode.toUpperCase()
  const configuredAgents = Array.from(
    { length: simConfig.numDrones },
    (_, id) => ({ id, alive: true }) as AgentPose,
  )
  const visibleAgents = agents.length ? agents : configuredAgents

  return (
    <section className="pybullet-sim" aria-label={`${missionName} PyBullet simulation`}>
      <div className="pybullet-sim__viewport">
        <canvas
          ref={canvasRef}
          className={`pybullet-sim__frame ${
            hasCanvasFrame && !frameSrc ? '' : 'pybullet-sim__frame--hidden'
          }`}
          aria-label={`${missionName} rendered by PyBullet`}
        />
        {frameSrc ? (
          <img
            className="pybullet-sim__frame"
            src={frameSrc}
            alt={`${missionName} rendered by PyBullet`}
          />
        ) : hasCanvasFrame ? null : (
          <div className="pybullet-sim__empty">
            <span>PYBULLET CAMERA</span>
            <strong>Waiting for rendered frames</strong>
          </div>
        )}
      </div>

      <div className="pybullet-video-treatment" aria-hidden="true" />

      <div className="pybullet-topline">
        <div className="pybullet-title-stack">
          <span>{scenario.label}</span>
          <strong>{scenario.name}</strong>
        </div>
        <div className="pybullet-live-cluster" aria-label="Live simulation status">
          <span>{modeLabel}</span>
          <span>{frameTime.toFixed(1)}s</span>
        </div>
      </div>

      <div className="pybullet-camera-controls" aria-label="Camera controls">
        {(['observer', 'chase', 'fpv'] as CameraMode[]).map((mode) => (
          <button
            key={mode}
            type="button"
            className={cameraMode === mode ? 'is-active' : ''}
            disabled={switchingCamera}
            onClick={() => void switchCamera(mode)}
          >
            {mode === 'fpv' ? 'FPV' : mode.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="pybullet-feed-rail" aria-label="Drone feeds">
        {visibleAgents.map((agent) => (
          <button
            key={agent.id}
            type="button"
            className={selectedDrone === agent.id ? 'is-active' : ''}
            disabled={switchingCamera}
            onClick={() => void switchCamera(cameraMode === 'observer' ? 'chase' : cameraMode, agent.id)}
          >
            <span>UAV-{String(agent.id + 1).padStart(2, '0')}</span>
            <i data-alive={agent.alive} />
          </button>
        ))}
      </div>

      <div className="pybullet-sidecar" aria-label="Mission telemetry">
        <div className="pybullet-stat-grid">
          <span>Policy {policy}</span>
          <span>Alive {alive || agents.length}/{simConfig.numDrones}</span>
          <span>Captures {captures}</span>
          <span>{contact ? 'CONTACT' : 'Searching'}</span>
        </div>

        <div className="pybullet-map" aria-label="Coordination map">
          {agents.map((agent) => (
            <i
              key={agent.id}
              data-alive={agent.alive}
              style={{
                left: `${Math.min(100, Math.max(0, (agent.x + 10) * 5))}%`,
                top: `${Math.min(100, Math.max(0, 100 - (agent.y + 10) * 5))}%`,
              }}
            />
          ))}
        </div>
      </div>

      <div className={`pybullet-fpv-overlay ${cameraMode === 'fpv' ? 'is-active' : ''}`} aria-hidden="true">
        <div className="pybullet-reticle">
          <i />
          <i />
          <i />
          <i />
        </div>
        <div className="pybullet-tape pybullet-tape--left">
          <span>ALT</span>
          <strong>{selectedAgent ? selectedAgent.z.toFixed(1) : '0.0'}</strong>
        </div>
        <div className="pybullet-tape pybullet-tape--right">
          <span>SPD</span>
          <strong>{speed.toFixed(1)}</strong>
        </div>
        <div className="pybullet-fpv-footer">
          <span>FEED UAV-{String(selectedDrone + 1).padStart(2, '0')}</span>
          <span>YAW {selectedAgent ? Math.round(selectedAgent.yaw * (180 / Math.PI)) : 0} DEG</span>
          <span>MAPPO ACTIVE</span>
        </div>
      </div>
    </section>
  )
}
