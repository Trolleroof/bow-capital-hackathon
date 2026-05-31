import { useEffect, useMemo, useRef, useState } from 'react'
import { PYBULLET_WS_URL } from '../gym/trainApi'

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
  agents: AgentPose[]
}

interface PyBulletFrameMessage {
  topic: 'pybullet_frame'
  t: number
  env_id?: string
  width: number
  height: number
  encoding: 'jpeg' | 'rgba'
  data: string
}

interface PyBulletSimPanelProps {
  envId: string
  missionName: string
  wsUrl?: string
}

type ConnectionState = 'connecting' | 'online' | 'offline'

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
  const [coverage, setCoverage] = useState(0)
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
          setFrameTime(message.t)
          return
        }

        if (message.topic === 'swarm') {
          setAgents(message.agents)
          setCoverage(message.coverage ?? 0)
          setPolicy(message.policy ?? 'trained')
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

      <div className="pybullet-hud pybullet-hud--left">
        <span className="pybullet-hud__kicker">Environment</span>
        <strong>{missionName}</strong>
        <div className="pybullet-stat-grid">
          <span data-state={connection}>WS {connection}</span>
          <span>Policy {policy}</span>
          <span>Alive {alive || agents.length}</span>
          <span>Frame {frameTime.toFixed(1)}s</span>
        </div>
      </div>

      <div className="pybullet-hud pybullet-hud--right">
        <span className="pybullet-hud__kicker">Coordination</span>
        <div className="pybullet-map" aria-hidden="true">
          {agents.map((agent) => (
            <i
              key={agent.id}
              style={{
                left: `${Math.min(100, Math.max(0, (agent.x + 10) * 5))}%`,
                top: `${Math.min(100, Math.max(0, 100 - (agent.y + 10) * 5))}%`,
              }}
            />
          ))}
        </div>
        <span className="pybullet-coverage">
          Coverage {(coverage * 100).toFixed(0)}%
        </span>
      </div>
    </section>
  )
}
