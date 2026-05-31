import { useState, useEffect, useCallback, useRef } from 'react'

const ORCH_WS = import.meta.env.VITE_COMBATOS_WS_URL ?? 'ws://localhost:8000'
const IMAGE_WS = import.meta.env.VITE_COMBATOS_IMAGE_WS_URL ?? 'ws://localhost:8001'
const CONTROL_TOPICS = [
  'pose',
  'detections',
  'recon',
  'slam_status',
  'slam_diagnostics',
] as const
const IMAGE_TOPICS = [
  'camera_frame',
  'fpv_raw',
  'slam_frame',
  'fpv_hud',
] as const

export interface SlamFrame {
  seq: number
  t: number
  width: number
  height: number
  source: string
  data: string
}

export interface SlamDiagnostics {
  droppedFrames: number
  cameraFrames: number
  annotatedFrames: number
  queueDepth: number
}

export interface Detection {
  id: string
  numericId: number
  cls: string
  conf: number
  rng: number
  brg: number
  bbox: [number, number, number, number] | null   // normalised [x,y,w,h] 0-1
  st: 'TRACK' | 'OBSERVE' | 'LOST'
  tone: 'amber' | 'candidate' | 'mute' | ''
  confirmed: boolean
}

export interface LogEntry {
  ts: string
  src: string
  tone: string
  msg: string
}

export interface TelemetryState {
  sec: number
  pose: { x: number; y: number; z: number }
  yaw: number
  vel: number
  drift: number
  slam: number
  yolo: number
  gpu: number
  temp: number
  heading: number
  loops: number
  traj: Array<{ x: number; y: number }>
  dets: Detection[]
  tracking: string
  gps: boolean
  recon: { status: 'training' | 'ready'; frames: number }
  wsConnected: boolean
  slamStatus: string
  cameraFrame: SlamFrame | null
  slamFrame: SlamFrame | null
  slamDiagnostics: SlamDiagnostics
}

function base64ToBlobUrl(data: string, mime = 'image/jpeg') {
  const binary = atob(data)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return URL.createObjectURL(new Blob([bytes], { type: mime }))
}

function fmtSec(s: number) {
  return 'T+' + String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0')
}

function initState(): TelemetryState {
  return {
    sec: 0,
    pose: { x: 0, y: 0, z: 0 },
    yaw: 0,
    vel: 0,
    drift: 0,
    slam: 0,
    yolo: 0,
    gpu: 0,
    temp: 0,
    heading: 0,
    loops: 0,
    traj: [],
    dets: [],
    tracking: '--',
    gps: false,
    recon: { status: 'training', frames: 0 },
    wsConnected: false,
    slamStatus: '--',
    cameraFrame: null,
    slamFrame: null,
    slamDiagnostics: {
      droppedFrames: 0,
      cameraFrames: 0,
      annotatedFrames: 0,
      queueDepth: 0,
    },
  }
}

function toFrame(msg: Record<string, unknown>): SlamFrame | null {
  if (typeof msg.data !== 'string') return null
  return {
    seq: typeof msg.seq === 'number' ? msg.seq : 0,
    t: typeof msg.t === 'number' ? msg.t : 0,
    width: typeof msg.width === 'number' ? msg.width : 0,
    height: typeof msg.height === 'number' ? msg.height : 0,
    source: typeof msg.source === 'string' ? msg.source : '',
    data: base64ToBlobUrl(msg.data),
  }
}

export function useCombatState() {
  const [t, setT] = useState<TelemetryState>(initState)
  const [log, setLog] = useState<LogEntry[]>([])

  // ws ref for sending confirm messages
  const wsRef = useRef<WebSocket | null>(null)

  const pushLog = useCallback((src: string, msg: string, tone = '') => {
    setT(p => {
      setLog(l => [...l.slice(-6), { ts: fmtSec(p.sec), src, msg, tone }])
      return p
    })
  }, [])

  // orchestrator WebSocket — real data when available
  useEffect(() => {
    let ws: WebSocket | null = null
    let retryTimeout: ReturnType<typeof setTimeout>

    function connect() {
      try {
        ws = new WebSocket(ORCH_WS)
        wsRef.current = ws

        ws.onopen = () => {
          ws?.send(JSON.stringify({
            type: 'subscribe',
            topics: CONTROL_TOPICS,
          }))
          setT(p => ({ ...p, wsConnected: true }))
          pushLog('BUS', 'orchestrator connected', 'amber')
        }

        ws.onclose = () => {
          setT(p => ({ ...p, wsConnected: false }))
          wsRef.current = null
          retryTimeout = setTimeout(connect, 3000)
        }

        ws.onerror = () => {
          ws?.close()
        }

        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data as string)
            if (msg.topic === 'pose') {
              const { x, y, z, qw = 1, qz = 0, tracking, gps = false } = msg
              const yaw = Math.atan2(2 * qw * qz, 1 - 2 * qz * qz) * (180 / Math.PI)
              setT(p => {
                const last = p.traj[p.traj.length - 1]
                const newPt = last
                  ? { x: x * 10 + last.x * 0.5, y: z * 10 + last.y * 0.5 }
                  : { x: x * 10, y: z * 10 }
                let traj = [...p.traj, newPt]
                if (traj.length > 46) traj = traj.slice(traj.length - 46)
                return {
                  ...p,
                  pose: { x, y, z },
                  yaw: (yaw + 360) % 360,
                  heading: (yaw + 360) % 360,
                  tracking: typeof tracking === 'string' ? tracking : p.tracking,
                  gps,
                  traj,
                }
              })
            } else if (msg.topic === 'detections') {
              const objects: Array<{
                id: number; cls: string; conf: number
                bbox: [number, number, number, number]
                is_primary: boolean; is_candidate: boolean; confirmed: boolean
              }> = (msg.objects as typeof objects) ?? []
              const mapped: Detection[] = objects.map((o) => ({
                id: `T-${String(o.id).padStart(4, '0')}`,
                numericId: o.id,
                cls: o.cls.toUpperCase(),
                conf: o.conf,
                rng: NaN,
                brg: NaN,
                bbox: Array.isArray(o.bbox) && o.bbox.length === 4 ? o.bbox : null,
                st: (o.confirmed || o.is_primary) ? 'TRACK' : 'OBSERVE',
                tone: o.is_primary ? 'amber' : o.is_candidate ? 'candidate' : '',
                confirmed: o.confirmed,
              }))
              setT(p => ({ ...p, dets: mapped }))
            } else if (msg.topic === 'recon') {
              setT(p => ({
                ...p,
                recon: {
                  status: msg.status === 'ready' ? 'ready' : 'training',
                  frames: msg.frames_used ?? p.recon.frames,
                },
              }))
              if (msg.status === 'ready') pushLog('RECON', '3DGS splat ready', 'amber')
            } else if (msg.topic === 'slam_status') {
              setT(p => {
                const tracking = typeof msg.tracking === 'string' ? msg.tracking : p.slamStatus
                return { ...p, tracking, slamStatus: tracking }
              })
            } else if (msg.topic === 'slam_diagnostics') {
              setT(p => ({
                ...p,
                slamDiagnostics: {
                  droppedFrames: msg.dropped_frames ?? p.slamDiagnostics.droppedFrames,
                  cameraFrames: msg.camera_frames ?? p.slamDiagnostics.cameraFrames,
                  annotatedFrames: msg.annotated_frames ?? p.slamDiagnostics.annotatedFrames,
                  queueDepth: msg.queue_depth ?? p.slamDiagnostics.queueDepth,
                },
              }))
            }
          } catch {
            // ignore parse errors
          }
        }
      } catch {
        retryTimeout = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      clearTimeout(retryTimeout)
      ws?.close()
      wsRef.current = null
    }
  }, [pushLog])

  useEffect(() => {
    let ws: WebSocket | null = null
    let retryTimeout: ReturnType<typeof setTimeout>

    function connect() {
      try {
        ws = new WebSocket(IMAGE_WS)

        ws.onopen = () => {
          ws?.send(JSON.stringify({
            type: 'subscribe',
            topics: IMAGE_TOPICS,
          }))
        }

        ws.onclose = () => {
          retryTimeout = setTimeout(connect, 3000)
        }

        ws.onerror = () => {
          ws?.close()
        }

        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data as string)
            if (msg.topic === 'camera_frame' || msg.topic === 'fpv_raw') {
              const frame = toFrame(msg)
              if (frame) {
                setT((p) => {
                  if (p.cameraFrame?.data.startsWith('blob:')) {
                    URL.revokeObjectURL(p.cameraFrame.data)
                  }
                  return { ...p, cameraFrame: frame }
                })
              }
            } else if (msg.topic === 'slam_frame' || msg.topic === 'fpv_hud') {
              const frame = toFrame(msg)
              if (frame) {
                setT((p) => {
                  if (p.slamFrame?.data.startsWith('blob:')) {
                    URL.revokeObjectURL(p.slamFrame.data)
                  }
                  return { ...p, slamFrame: frame }
                })
              }
            }
          } catch {
            // ignore parse errors
          }
        }
      } catch {
        retryTimeout = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      clearTimeout(retryTimeout)
      ws?.close()
      setT((p) => {
        if (p.cameraFrame?.data.startsWith('blob:')) {
          URL.revokeObjectURL(p.cameraFrame.data)
        }
        if (p.slamFrame?.data.startsWith('blob:')) {
          URL.revokeObjectURL(p.slamFrame.data)
        }
        return p
      })
    }
  }, [])

  const _sendCmd = useCallback((payload: object) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ topic: 'command', ...payload }))
    }
  }, [])

  const followTarget = useCallback((numericId: number, id: string) => {
    _sendCmd({ action: 'follow', track_id: numericId })
    setT(p => ({
      ...p,
      dets: p.dets.map(d => ({ ...d, tone: d.numericId === numericId ? 'amber' : d.tone })),
    }))
    pushLog('CMD', `follow → ${id}`, 'amber')
  }, [_sendCmd, pushLog])

  const confirmTarget = useCallback((numericId: number, id: string) => {
    _sendCmd({ action: 'confirm', track_id: numericId })
    setT(p => ({
      ...p,
      dets: p.dets.map(d => ({ ...d, confirmed: d.numericId === numericId })),
    }))
    pushLog('CMD', `confirmed ${id}`, 'amber')
  }, [_sendCmd, pushLog])

  const releaseTarget = useCallback(() => {
    _sendCmd({ action: 'release' })
    setT(p => {
      const hasConfirmed = p.dets.some(d => d.confirmed)
      if (hasConfirmed) {
        // confirmed → followed: clear confirmed flag
        return { ...p, dets: p.dets.map(d => ({ ...d, confirmed: false })) }
      } else {
        // followed → proposed: clear amber
        return { ...p, dets: p.dets.map(d => ({ ...d, tone: (d.tone === 'amber' ? '' : d.tone) as Detection['tone'] })) }
      }
    })
    pushLog('CMD', 'released', '')
  }, [_sendCmd, pushLog])

  return { t, log, followTarget, confirmTarget, releaseTarget }
}
