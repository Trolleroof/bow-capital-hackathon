import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

const AMBER = 0x3fa3e6
const AMBER_BR = 0x74c8ff
const GREEN = 0x37b8a4
const METAL = 0x121a26

function buildScene(canvas: HTMLCanvasElement) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
  renderer.setClearColor(0x000000, 0)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

  const scene = new THREE.Scene()
  scene.fog = new THREE.FogExp2(0x070a10, 0.045)

  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100)
  camera.position.set(0, 3.4, 8.6)
  camera.lookAt(0, 0.45, 0)

  scene.add(new THREE.AmbientLight(0x22272f, 0.9))
  const key = new THREE.PointLight(AMBER_BR, 1.5, 40); key.position.set(4, 7, 6); scene.add(key)
  const rim = new THREE.PointLight(GREEN, 0.5, 40); rim.position.set(-7, 2, -5); scene.add(rim)
  const fill = new THREE.DirectionalLight(0xa8bcd0, 0.35); fill.position.set(-3, 4, 8); scene.add(fill)

  const bodyMat = new THREE.MeshStandardMaterial({ color: METAL, metalness: 0.55, roughness: 0.42 })
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x0b1019, metalness: 0.4, roughness: 0.7 })
  const edgeMat = new THREE.LineBasicMaterial({ color: AMBER, transparent: true, opacity: 0.85 })
  const opticMat = new THREE.MeshStandardMaterial({ color: AMBER, emissive: AMBER, emissiveIntensity: 1.4, metalness: 0.3, roughness: 0.3 })
  const bladeMat = new THREE.MeshStandardMaterial({ color: 0x1a2230, metalness: 0.5, roughness: 0.5, transparent: true, opacity: 0.9 })
  const discMat = new THREE.MeshBasicMaterial({ color: AMBER, transparent: true, opacity: 0.06, side: THREE.DoubleSide })

  function edged(geo: THREE.BufferGeometry, parent: THREE.Object3D) {
    const eg = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 18), edgeMat)
    parent.add(eg)
  }

  const drone = new THREE.Group()
  scene.add(drone)

  const hullGeo = new THREE.BoxGeometry(2.0, 0.5, 2.0)
  const hull = new THREE.Mesh(hullGeo, bodyMat); edged(hullGeo, hull); drone.add(hull)

  const deckGeo = new THREE.BoxGeometry(1.3, 0.34, 1.3)
  const deck = new THREE.Mesh(deckGeo, bodyMat); deck.position.y = 0.4; edged(deckGeo, deck); drone.add(deck)

  const canopyGeo = new THREE.CylinderGeometry(0.34, 0.5, 0.4, 6)
  const canopy = new THREE.Mesh(canopyGeo, darkMat); canopy.position.y = 0.72; canopy.rotation.y = Math.PI / 6
  edged(canopyGeo, canopy); drone.add(canopy)

  const gimbal = new THREE.Mesh(new THREE.SphereGeometry(0.32, 18, 14), opticMat)
  gimbal.position.y = -0.34; drone.add(gimbal)
  const gimbalRing = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.05, 8, 24), darkMat)
  gimbalRing.position.y = -0.2; gimbalRing.rotation.x = Math.PI / 2; drone.add(gimbalRing)

  const rotors: THREE.Group[] = []
  const armAngles = [Math.PI / 4, 3 * Math.PI / 4, 5 * Math.PI / 4, 7 * Math.PI / 4]

  armAngles.forEach(a => {
    const ax = Math.cos(a), az = Math.sin(a), reach = 1.9

    const armGeo = new THREE.BoxGeometry(0.26, 0.16, 2.6)
    const arm = new THREE.Mesh(armGeo, bodyMat)
    arm.position.set(ax * 1.0, 0, az * 1.0); arm.rotation.y = -a + Math.PI / 2
    edged(armGeo, arm); drone.add(arm)

    const mx = ax * reach, mz = az * reach
    const motorGeo = new THREE.CylinderGeometry(0.3, 0.34, 0.5, 14)
    const motor = new THREE.Mesh(motorGeo, darkMat)
    motor.position.set(mx, 0.06, mz); edged(motorGeo, motor); drone.add(motor)

    const rotor = new THREE.Group()
    rotor.position.set(mx, 0.34, mz)
    for (let b = 0; b < 2; b++) {
      const bladeGeo = new THREE.BoxGeometry(2.0, 0.04, 0.18)
      const blade = new THREE.Mesh(bladeGeo, bladeMat)
      blade.rotation.y = b * Math.PI / 2; rotor.add(blade)
    }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(1.05, 28), discMat)
    disc.rotation.x = -Math.PI / 2; rotor.add(disc)
    drone.add(rotor); rotors.push(rotor)

    const strutGeo = new THREE.BoxGeometry(0.08, 0.7, 0.08)
    const strut = new THREE.Mesh(strutGeo, darkMat)
    strut.position.set(mx * 0.6, -0.45, mz * 0.6); drone.add(strut)
  });

  [-1, 1].forEach(side => {
    const skidGeo = new THREE.BoxGeometry(0.1, 0.1, 2.8)
    const skid = new THREE.Mesh(skidGeo, darkMat)
    skid.position.set(side * 1.1, -0.86, 0); edged(skidGeo, skid); drone.add(skid)
  })

  const reticle = new THREE.Mesh(
    new THREE.TorusGeometry(3.6, 0.012, 6, 80),
    new THREE.MeshBasicMaterial({ color: AMBER, transparent: true, opacity: 0.22 })
  )
  reticle.rotation.x = Math.PI / 2; reticle.position.y = -0.2; scene.add(reticle)

  const reticle2 = new THREE.Mesh(
    new THREE.TorusGeometry(2.9, 0.01, 6, 80),
    new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0.14 })
  )
  reticle2.rotation.x = Math.PI / 2; reticle2.position.y = -0.2; scene.add(reticle2)

  const grid = new THREE.GridHelper(40, 40, 0x274463, 0x16202e)
  grid.position.y = -1.7;
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.5; scene.add(grid)

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight
    if (!w || !h) return
    renderer.setSize(w, h, false)
    camera.aspect = w / h; camera.updateProjectionMatrix()
  }
  resize()

  const state = { mx: 0, my: 0, launch: 0, raf: 0, t: 0 }

  function loop() {
    state.t += 0.016
    const spin = 0.6 + state.launch * 2.6
    rotors.forEach((r, i) => r.rotation.y += spin * (i % 2 ? 1 : -1))
    drone.rotation.y += 0.0035
    drone.position.y = Math.sin(state.t * 1.4) * 0.08 + state.launch * 2.2
    reticle.rotation.z += 0.004; reticle2.rotation.z -= 0.006
    const tx = state.mx * 1.3, ty = 3.4 + state.my * 0.8 - state.launch * 0.6
    camera.position.x += (tx - camera.position.x) * 0.05
    camera.position.y += (ty - camera.position.y) * 0.05
    camera.position.z += ((8.6 - state.launch * 3.2) - camera.position.z) * 0.05
    camera.lookAt(0, 0.45 + state.launch * 1.4, 0)
    renderer.render(scene, camera)
    state.raf = requestAnimationFrame(loop)
  }
  state.raf = requestAnimationFrame(loop)

  return {
    state, resize,
    dispose() { cancelAnimationFrame(state.raf); renderer.dispose() },
  }
}

export function BootView({ onLaunch }: { onLaunch: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<ReturnType<typeof buildScene> | null>(null)
  const [launching, setLaunching] = useState(false)

  useEffect(() => {
    if (!canvasRef.current) return
    const inst = buildScene(canvasRef.current)
    sceneRef.current = inst

    const onResize = () => inst.resize()
    const onMove = (e: MouseEvent) => {
      inst.state.mx = (e.clientX / window.innerWidth - 0.5) * 2
      inst.state.my = (e.clientY / window.innerHeight - 0.5) * 2
    }
    window.addEventListener('resize', onResize)
    window.addEventListener('mousemove', onMove)
    return () => {
      window.removeEventListener('resize', onResize)
      window.removeEventListener('mousemove', onMove)
      inst.dispose()
    }
  }, [])

  const launch = () => {
    if (launching) return
    setLaunching(true)
    const inst = sceneRef.current
    if (inst) {
      const t0 = performance.now()
      const ramp = () => {
        const p = Math.min(1, (performance.now() - t0) / 950)
        inst.state.launch = p * p
        if (p < 1) requestAnimationFrame(ramp)
      }
      ramp()
    }
    setTimeout(onLaunch, 1050)
  }

  return (
    <div className={'boot' + (launching ? ' is-launching' : '')}>
      <canvas className="boot-canvas" ref={canvasRef} />
      <div className="boot-vign" />
      <div className="boot-sweepline" />

      <div className="bfr tl" /><div className="bfr tr" />
      <div className="bfr bl" /><div className="bfr br" />

      <div className="boot-title">
        <h1>COMBAT<span>OS</span></h1>
        <div className="bt-sub">GROUND CONTROL STATION — EDGE AUTONOMY STACK</div>
      </div>

      <button className="boot-enter" onClick={launch} disabled={launching}>
        <span className="be-lbl">{launching ? 'Starting…' : 'Enter'}</span>
        <span className="be-arrow">{launching ? '◢◤' : '▸'}</span>
      </button>
      <div className="boot-hint">CLICK TO ENTER · DRAG TO ORBIT</div>
    </div>
  )
}
