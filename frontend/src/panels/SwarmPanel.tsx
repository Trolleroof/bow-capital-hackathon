/**
 * SwarmPanel — Phase 0 Three.js view of the CombatOS swarm.
 *
 * Connects to the Python WebSocket bus (swarm/bus.py), subscribes to the `swarm`
 * topic, and renders N drones as dots moving inside a bounded box. Shows a
 * `COMMS: DENIED` label and the live agent count.
 *
 * Bus message (SWARM.md §4 / TEAM_PLAN §5):
 *   { "topic": "swarm", "t": 1234.56, "comms": "denied",
 *     "agents": [ { id, x, y, z, yaw, role, alive }, ... ] }
 *
 * Self-contained: just render <SwarmPanel /> anywhere. Default bus URL is
 * ws://localhost:8765 (override via the `url` prop).
 */
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

// --- bus message types -------------------------------------------------------
interface SwarmAgent {
  id: number
  x: number
  y: number
  z: number
  yaw: number
  role: string
  alive: boolean
}
interface SwarmMessage {
  topic?: string
  t: number
  comms: string
  agents: SwarmAgent[]
}

// World spans [-WORLD_HALF, WORLD_HALF] in x and y (matches env.WORLD_HALF).
const WORLD_HALF = 10

interface SwarmPanelProps {
  url?: string
}

export default function SwarmPanel({ url = 'ws://localhost:8765' }: SwarmPanelProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  // latest message, written by the WS handler and read by the render loop
  const latest = useRef<SwarmMessage | null>(null)
  const [connected, setConnected] = useState(false)
  const [nAlive, setNAlive] = useState(0)
  const [comms, setComms] = useState('—')

  // --- WebSocket subscription ---
  useEffect(() => {
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let closed = false

    const connect = () => {
      ws = new WebSocket(url)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = setTimeout(connect, 1000)
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as SwarmMessage
          if (msg.topic && msg.topic !== 'swarm') return
          if (!Array.isArray(msg.agents)) return
          latest.current = msg
          setComms(msg.comms ?? '—')
          setNAlive(msg.agents.filter((a) => a.alive).length)
        } catch {
          /* ignore malformed frames */
        }
      }
    }
    connect()
    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      ws?.close()
    }
  }, [url])

  // --- Three.js scene ---
  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const width = mount.clientWidth || 480
    const height = mount.clientHeight || 360

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0e12)

    // top-down-ish ortho-style perspective camera looking at the world box
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000)
    camera.position.set(0, WORLD_HALF * 2.2, WORLD_HALF * 1.6)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(width, height)
    mount.appendChild(renderer.domElement)

    scene.add(new THREE.AmbientLight(0xffffff, 0.8))
    const dir = new THREE.DirectionalLight(0xffffff, 0.6)
    dir.position.set(5, 20, 10)
    scene.add(dir)

    // bounded world: ground grid + edge box
    const grid = new THREE.GridHelper(WORLD_HALF * 2, 20, 0x1f6f4f, 0x16313a)
    scene.add(grid)
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(
        new THREE.BoxGeometry(WORLD_HALF * 2, 6, WORLD_HALF * 2),
      ),
      new THREE.LineBasicMaterial({ color: 0x2fae7a }),
    )
    box.position.y = 3
    scene.add(box)

    // drone meshes, created lazily / reused across frames
    const droneGeo = new THREE.SphereGeometry(0.5, 16, 16)
    const aliveMat = new THREE.MeshStandardMaterial({
      color: 0x4ef0a0,
      emissive: 0x123b2a,
    })
    const deadMat = new THREE.MeshStandardMaterial({
      color: 0x553333,
      emissive: 0x000000,
    })
    const drones: THREE.Mesh[] = []

    // map env world coords (x,y,z) -> three coords. env y is the ground plane,
    // env z is altitude -> three's vertical axis.
    const toScene = (a: SwarmAgent) =>
      new THREE.Vector3(a.x, a.z, -a.y)

    let raf = 0
    const animate = () => {
      raf = requestAnimationFrame(animate)
      const msg = latest.current
      if (msg) {
        for (let i = 0; i < msg.agents.length; i++) {
          const a = msg.agents[i]
          let m = drones[i]
          if (!m) {
            m = new THREE.Mesh(droneGeo, aliveMat)
            drones[i] = m
            scene.add(m)
          }
          m.position.copy(toScene(a))
          m.material = a.alive ? aliveMat : deadMat
          m.scale.setScalar(a.alive ? 1 : 0.6)
        }
        // hide any stale extra meshes (agent count shrank)
        for (let i = msg.agents.length; i < drones.length; i++) {
          drones[i].visible = false
        }
      }
      renderer.render(scene, camera)
    }
    animate()

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
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      droneGeo.dispose()
      aliveMat.dispose()
      deadMat.dispose()
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement)
      }
    }
  }, [])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 360 }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          fontFamily: 'monospace',
          color: '#4ef0a0',
          pointerEvents: 'none',
          textShadow: '0 0 4px #000',
        }}
      >
        <div style={{ fontWeight: 700, letterSpacing: 1 }}>
          COMMS: {comms.toUpperCase()}
        </div>
        <div style={{ opacity: 0.85 }}>AGENTS ALIVE: {nAlive}</div>
        <div style={{ opacity: 0.6, fontSize: 11 }}>
          {connected ? 'bus connected' : 'connecting…'}
        </div>
      </div>
    </div>
  )
}
