import { useState, useEffect, useCallback, useRef } from 'react'

const ORCH_WS = 'ws://localhost:8000'

export interface Detection {
  id: string
  cls: string
  conf: number
  rng: number
  brg: number
  st: 'TRACK' | 'OBSERVE' | 'LOST'
  tone: 'amber' | 'mute' | ''
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

const MOCK_DETS: Detection[] = [
  { id: 'TGT-01', cls: 'SUBJECT', conf: 0.94, rng: 42.1, brg: 14,  st: 'TRACK',   tone: 'amber', confirmed: false },
  { id: 'E-2207', cls: 'PERSON',  conf: 0.71, rng: 58.3, brg: 331, st: 'OBSERVE', tone: '',      confirmed: false },
  { id: 'E-2208', cls: 'PERSON',  conf: 0.66, rng: 61.0, brg: 337, st: 'OBSERVE', tone: '',      confirmed: false },
  { id: 'V-0714', cls: 'VEHICLE', conf: 0.82, rng: 88.4, brg: 48,  st: 'OBSERVE', tone: '',      confirmed: false },
  { id: 'E-2209', cls: 'PERSON',  conf: 0.58, rng: 73.9, brg: 9,   st: 'LOST',    tone: 'mute',  confirmed: false },
]

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
    dets: MOCK_DETS.map(d => ({ ...d })),
    tracking: 'OK',
    gps: false,
    recon: { status: 'training', frames: 0 },
    wsConnected: false,
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
          dets: p.dets.map(d =>
            d.st === 'LOST' ? d : {
              ...d,
              conf: Math.min(0.99, Math.max(0.4, +(d.conf + (Math.random() - 0.5) * 0.02).toFixed(2))),
              rng: +(d.rng + (Math.random() - 0.5) * 0.6).toFixed(1),
            }
          ),
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
                bbox: [number, number, number, number]; is_target: boolean; confirmed: boolean
              }> = msg.objects ?? []
              if (objects.length > 0) {
                const mapped: Detection[] = objects.map((o, i) => ({
                  id: `T-${String(o.id).padStart(4, '0')}`,
                  cls: o.cls.toUpperCase(),
                  conf: o.conf,
                  rng: NaN,
                  brg: NaN,
                  st: o.is_target ? 'TRACK' : 'OBSERVE',
                  tone: o.is_target ? 'amber' : '',
                  confirmed: o.confirmed,
                }))
                setT(p => ({ ...p, dets: mapped }))
              }
            } else if (msg.topic === 'recon') {
              setT(p => ({
                ...p,
                recon: {
                  status: msg.status === 'ready' ? 'ready' : 'training',
                  frames: msg.frames_used ?? p.recon.frames,
                },
              }))
              if (msg.status === 'ready') pushLog('RECON', '3DGS splat ready', 'amber')
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

  const confirmTarget = useCallback((id: string) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ topic: 'confirm_target', id }))
    }
    setT(p => ({
      ...p,
      dets: p.dets.map(d => d.id === id ? { ...d, confirmed: true } : d),
    }))
    pushLog('CMD', `target ${id} confirmed`, 'amber')
  }, [pushLog])

  return { t, log, confirmTarget }
}
