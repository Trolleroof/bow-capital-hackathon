import { useState, useEffect, useCallback, useRef } from 'react'

const ORCH_WS = 'ws://localhost:8000'

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

function j(v: number, a: number) { return +(v + (Math.random() - 0.5) * a) }

function makeTraj(n = 24) {
  const pts: Array<{ x: number; y: number }> = []
  let x = 0, y = 0
  for (let i = 0; i < n; i++) {
    x += 5 + Math.random() * 2.5
    y += Math.sin(i * 0.55) * 3.4 + (Math.random() - 0.42) * 4.5
    pts.push({ x, y })
  }
  return pts
}

function fmtSec(s: number) {
  return 'T+' + String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0')
}

function initState(): TelemetryState {
  return {
    sec: 137,
    pose: { x: 12.84, y: -3.10, z: 1.62 },
    yaw: 147.3,
    vel: 2.41,
    drift: 0.8,
    slam: 28,
    yolo: 19,
    gpu: 81,
    temp: 64,
    heading: 150,
    loops: 14,
    traj: makeTraj(24),
    dets: [],
    tracking: 'OK',
    gps: false,
    recon: { status: 'training', frames: 0 },
    wsConnected: false,
    slamStatus: 'NO_LOCK',
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
    data: `data:image/jpeg;base64,${msg.data}`,
  }
}

export function useCombatState() {
  const [t, setT] = useState<TelemetryState>(initState)
  const [log, setLog] = useState<LogEntry[]>([
    { ts: 'T+02:14', src: 'NAV',   tone: '', msg: 'loop closure accepted #14' },
    { ts: 'T+02:16', src: 'NAV',   tone: '', msg: 'VSLAM tracking nominal' },
    { ts: 'T+02:17', src: 'RECON', tone: '', msg: 'splat asset cached' },
  ])

  // ws ref for sending confirm messages
  const wsRef = useRef<WebSocket | null>(null)

  const pushLog = useCallback((src: string, msg: string, tone = '') => {
    setT(p => {
      setLog(l => [...l.slice(-6), { ts: fmtSec(p.sec), src, msg, tone }])
      return p
    })
  }, [])

  // mock simulation tick
  useEffect(() => {
    const id = setInterval(() => {
      setT(p => {
        const last = p.traj[p.traj.length - 1]
        let traj = [
          ...p.traj,
          {
            x: last.x + 5 + Math.random() * 2.5,
            y: last.y + Math.sin(p.sec * 0.4) * 3 + (Math.random() - 0.42) * 4.5,
          },
        ]
        if (traj.length > 46) traj = traj.slice(traj.length - 46)

        return {
          ...p,
          sec: p.sec + 1,
          pose: {
            x: j(12.84 + (p.sec * 0.015 % 6), 0.05),
            y: j(-3.10, 0.05),
            z: j(1.62, 0.03),
          },
          yaw: ((p.yaw + (Math.random() - 0.5) * 1.4) + 360) % 360,
          vel: Math.max(0.6, j(2.41, 0.16)),
          drift: Math.min(1.8, Math.max(0.4, j(p.drift, 0.07))),
          slam: Math.round(j(28, 1.5)),
          yolo: Math.round(j(19, 1.7)),
          gpu: Math.round(Math.min(96, Math.max(62, j(81, 3)))),
          temp: Math.round(j(64, 0.7)),
          heading: (p.heading + 0.7) % 360,
          traj,
          dets: p.dets,
        }
      })
    }, 850)
    return () => clearInterval(id)
  }, [])

  // mock log ticker
  useEffect(() => {
    const pool: [string, string, string][] = [
      ['NAV', 'keyframe inserted', ''],
      ['NAV', 'VSLAM tracking nominal', ''],
      ['NAV', 'IMU pre-integration ok', ''],
      ['RECON', 'fly-through armed', ''],
    ]
    const id = setInterval(() => {
      const e = pool[Math.floor(Math.random() * pool.length)]
      pushLog(e[0], e[1], e[2])
    }, 4200)
    return () => clearInterval(id)
  }, [pushLog])

  // orchestrator WebSocket — real data when available
  useEffect(() => {
    let ws: WebSocket | null = null
    let retryTimeout: ReturnType<typeof setTimeout>

    function connect() {
      try {
        ws = new WebSocket(ORCH_WS)
        wsRef.current = ws

        ws.onopen = () => {
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
              const { x, y, z, qw = 1, qz = 0, tracking = 'OK', gps = false } = msg
              const yaw = Math.atan2(2 * qw * qz, 1 - 2 * qz * qz) * (180 / Math.PI)
              setT(p => {
                const last = p.traj[p.traj.length - 1]
                const newPt = { x: x * 10 + last.x * 0.5, y: z * 10 + last.y * 0.5 }
                let traj = [...p.traj, newPt]
                if (traj.length > 46) traj = traj.slice(traj.length - 46)
                return {
                  ...p,
                  pose: { x, y, z },
                  yaw: (yaw + 360) % 360,
                  heading: (yaw + 360) % 360,
                  tracking,
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
              const tracking = typeof msg.tracking === 'string' ? msg.tracking : 'NO_LOCK'
              setT(p => ({ ...p, tracking, slamStatus: tracking }))
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
            } else if (msg.topic === 'camera_frame' || msg.topic === 'fpv_raw') {
              const frame = toFrame(msg)
              if (frame) setT(p => ({ ...p, cameraFrame: frame }))
            } else if (msg.topic === 'slam_frame' || msg.topic === 'fpv_hud') {
              const frame = toFrame(msg)
              if (frame) setT(p => ({ ...p, slamFrame: frame }))
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
