import { useEffect, useMemo, useState } from 'react'

interface PoseMessage {
  topic: 'pose'
  t: number
  x: number
  y: number
  z: number
  qx?: number
  qy?: number
  qz?: number
  qw?: number
  tracking?: string
}

interface SlamStatusMessage {
  topic: 'slam_status'
  t: number
  tracking?: string
  connected?: boolean
  camera_hz?: number
  annotated_hz?: number
  dropped_frames?: number
}

interface SlamDiagnosticsMessage {
  topic: 'slam_diagnostics'
  t: number
  tracking?: string
  dropped_frames?: number
  camera_frames?: number
  annotated_frames?: number
  queue_depth?: number
}

interface SlamFrameMessage {
  topic: 'camera_frame' | 'camera_right_frame' | 'slam_frame'
  t: number
  width: number
  height: number
  seq: number
  source?: string
  encoding: 'jpeg'
  data: string
}

interface StatusMessage {
  topic: 'status'
  localized?: boolean
  modules?: { nav?: string }
}

type BusMessage = PoseMessage | SlamStatusMessage | SlamDiagnosticsMessage | SlamFrameMessage | StatusMessage

const BUS_URL = import.meta.env.VITE_COMBATOS_WS_URL ?? 'ws://localhost:8000'
const IMAGE_URL = import.meta.env.VITE_COMBATOS_IMAGE_WS_URL ?? 'ws://localhost:8001'

function base64ToBlobUrl(data: string, mime = 'image/jpeg') {
  const binary = atob(data)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return URL.createObjectURL(new Blob([bytes], { type: mime }))
}

function fmt(value: number | undefined, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '--'
}

export default function SlamTestPanel() {
  const [connected, setConnected] = useState(false)
  const [pose, setPose] = useState<PoseMessage | null>(null)
  const [status, setStatus] = useState<SlamStatusMessage | null>(null)
  const [diagnostics, setDiagnostics] = useState<SlamDiagnosticsMessage | null>(null)
  const [systemStatus, setSystemStatus] = useState<StatusMessage | null>(null)
  const [cameraFrame, setCameraFrame] = useState<SlamFrameMessage | null>(null)
  const [slamFrame, setSlamFrame] = useState<SlamFrameMessage | null>(null)

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let closed = false

    const connect = () => {
      ws = new WebSocket(BUS_URL)
      ws.onopen = () => {
        setConnected(true)
        ws?.send(JSON.stringify({
          type: 'subscribe',
          topics: ['pose', 'status', 'slam_status', 'slam_diagnostics'],
        }))
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = setTimeout(connect, 1000)
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as BusMessage
          if (msg.topic === 'pose') setPose(msg)
          else if (msg.topic === 'slam_status') setStatus(msg)
          else if (msg.topic === 'slam_diagnostics') setDiagnostics(msg)
          else if (msg.topic === 'status') setSystemStatus(msg)
        } catch {
          // Ignore malformed test frames.
        }
      }
    }

    connect()
    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      ws?.close()
    }
  }, [])

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let closed = false

    const connect = () => {
      ws = new WebSocket(IMAGE_URL)
      ws.onopen = () => {
        ws?.send(JSON.stringify({
          type: 'subscribe',
          topics: ['camera_frame', 'slam_frame'],
        }))
      }
      ws.onclose = () => {
        if (!closed) retry = setTimeout(connect, 1000)
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as BusMessage
          if (msg.topic === 'camera_frame') {
            setCameraFrame((prev) => {
              if (prev?.data.startsWith('blob:')) {
                URL.revokeObjectURL(prev.data)
              }
              return { ...msg, data: base64ToBlobUrl(msg.data) }
            })
          } else if (msg.topic === 'slam_frame') {
            setSlamFrame((prev) => {
              if (prev?.data.startsWith('blob:')) {
                URL.revokeObjectURL(prev.data)
              }
              return { ...msg, data: base64ToBlobUrl(msg.data) }
            })
          }
        } catch {
          // Ignore malformed test frames.
        }
      }
    }

    connect()
    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      ws?.close()
      setCameraFrame((prev) => {
        if (prev?.data.startsWith('blob:')) {
          URL.revokeObjectURL(prev.data)
        }
        return prev
      })
      setSlamFrame((prev) => {
        if (prev?.data.startsWith('blob:')) {
          URL.revokeObjectURL(prev.data)
        }
        return prev
      })
    }
  }, [])

  const cameraSrc = useMemo(() => cameraFrame?.data ?? '', [cameraFrame])
  const slamSrc = useMemo(() => slamFrame?.data ?? '', [slamFrame])
  const tracking = status?.tracking ?? pose?.tracking ?? diagnostics?.tracking ?? 'NO_LOCK'
  const navState = systemStatus?.modules?.nav ?? 'down'

  return (
    <aside className="slam-test-panel" aria-label="SLAM test stream">
      <div className="slam-test-head">
        <strong>SLAM TEST</strong>
        <span className={connected ? 'is-up' : 'is-down'}>{connected ? 'BUS UP' : 'BUS DOWN'}</span>
      </div>

      <div className="slam-test-video-grid">
        <figure>
          <div className="slam-test-video">
            {cameraSrc ? <img src={cameraSrc} alt="OAK left camera stream" /> : <span>Waiting for camera_frame</span>}
          </div>
          <figcaption>Camera {cameraFrame ? `#${cameraFrame.seq}` : ''}</figcaption>
        </figure>
        <figure>
          <div className="slam-test-video">
            {slamSrc ? <img src={slamSrc} alt="Annotated ORB-SLAM frame" /> : <span>Waiting for slam_frame</span>}
          </div>
          <figcaption>Annotated {slamFrame ? `#${slamFrame.seq}` : ''}</figcaption>
        </figure>
      </div>

      <div className="slam-test-metrics">
        <span>Tracking: <b>{tracking}</b></span>
        <span>Nav: <b>{navState}</b></span>
        <span>Localized: <b>{systemStatus?.localized ? 'true' : 'false'}</b></span>
        <span>Pose: <b>{fmt(pose?.x)}, {fmt(pose?.y)}, {fmt(pose?.z)}</b></span>
        <span>Dropped: <b>{diagnostics?.dropped_frames ?? status?.dropped_frames ?? 0}</b></span>
        <span>Queue: <b>{diagnostics?.queue_depth ?? 0}</b></span>
      </div>
    </aside>
  )
}
