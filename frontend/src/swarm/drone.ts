/**
 * drone.ts — Three.js visuals for the swarm: a quadrotor-ish drone mesh and a
 * per-drone fading motion trail. Kept out of SwarmPanel so the React component
 * stays focused on the data/render loop.
 */
import * as THREE from 'three'

const ARM_LEN = 0.68
const ROTOR_R = 0.3

// Shared geometries/materials (created once, reused for every drone).
const bodyGeo = new THREE.BoxGeometry(0.58, 0.18, 0.48)
const canopyGeo = new THREE.BoxGeometry(0.36, 0.12, 0.28)
const armGeo = new THREE.BoxGeometry(ARM_LEN * 2, 0.045, 0.07)
const motorGeo = new THREE.CylinderGeometry(0.11, 0.12, 0.12, 20)
const rotorHubGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.06, 16)
const rotorDiscGeo = new THREE.CylinderGeometry(ROTOR_R, ROTOR_R, 0.012, 40, 1, true)
const rotorRingGeo = new THREE.TorusGeometry(ROTOR_R, 0.012, 8, 40)
const cameraGeo = new THREE.CylinderGeometry(0.055, 0.07, 0.1, 16)

const aliveBody = new THREE.MeshStandardMaterial({
  color: 0x25322f,
  emissive: 0x07110e,
  emissiveIntensity: 0.45,
  metalness: 0.55,
  roughness: 0.34,
})
const aliveTrim = new THREE.MeshStandardMaterial({
  color: 0x78d7b2,
  emissive: 0x13523d,
  emissiveIntensity: 0.85,
  metalness: 0.25,
  roughness: 0.38,
})
const aliveRotorDisc = new THREE.MeshBasicMaterial({
  color: 0xc4fff0,
  transparent: true,
  opacity: 0.2,
  depthWrite: false,
  side: THREE.DoubleSide,
})
const aliveRotorRing = new THREE.MeshBasicMaterial({
  color: 0x8ff4c7,
  transparent: true,
  opacity: 0.72,
})
const deadBody = new THREE.MeshStandardMaterial({
  color: 0x2a1414,
  emissive: 0x000000,
  metalness: 0.1,
  roughness: 0.9,
})
const deadRotor = new THREE.MeshStandardMaterial({
  color: 0x4a2a2a,
  emissive: 0x000000,
})
const deadRotorDisc = new THREE.MeshBasicMaterial({
  color: 0x6f4747,
  transparent: true,
  opacity: 0.12,
  depthWrite: false,
  side: THREE.DoubleSide,
})

export interface Drone {
  group: THREE.Group
  rotors: THREE.Group[]
  body: THREE.Mesh
  arms: THREE.Mesh[]
  trim: THREE.Mesh[]
  rotorDiscs: THREE.Mesh[]
  rotorRings: THREE.Mesh[]
  /** spin-down state for dead drones */
  spin: number
}

/** Build one reusable quadrotor: a body + two crossed arms + 4 rotor discs. */
export function makeDrone(): Drone {
  const group = new THREE.Group()
  group.scale.setScalar(0.78)

  const body = new THREE.Mesh(bodyGeo, aliveBody)
  body.castShadow = true
  group.add(body)

  const canopy = new THREE.Mesh(canopyGeo, aliveTrim)
  canopy.position.y = 0.13
  canopy.position.z = -0.05
  group.add(canopy)

  const camera = new THREE.Mesh(cameraGeo, aliveTrim)
  camera.rotation.x = Math.PI / 2
  camera.position.set(0, -0.005, -0.3)
  group.add(camera)

  const arm1 = new THREE.Mesh(armGeo, aliveBody)
  const arm2 = new THREE.Mesh(armGeo, aliveBody)
  arm2.rotation.y = Math.PI / 2
  group.add(arm1, arm2)

  const rotors: THREE.Group[] = []
  const rotorDiscs: THREE.Mesh[] = []
  const rotorRings: THREE.Mesh[] = []
  const trim: THREE.Mesh[] = [canopy, camera]
  const offs: [number, number][] = [
    [ARM_LEN, ARM_LEN],
    [-ARM_LEN, ARM_LEN],
    [ARM_LEN, -ARM_LEN],
    [-ARM_LEN, -ARM_LEN],
  ]
  for (const [dx, dz] of offs) {
    const rotor = new THREE.Group()
    rotor.position.set(dx, 0.08, dz)

    const motor = new THREE.Mesh(motorGeo, aliveBody)
    const hub = new THREE.Mesh(rotorHubGeo, aliveTrim)
    const disc = new THREE.Mesh(rotorDiscGeo, aliveRotorDisc)
    const ring = new THREE.Mesh(rotorRingGeo, aliveRotorRing)
    motor.rotation.x = Math.PI / 2
    hub.rotation.x = Math.PI / 2
    disc.rotation.x = Math.PI / 2
    ring.rotation.x = Math.PI / 2

    rotor.add(motor, hub, disc, ring)
    group.add(rotor)
    rotors.push(rotor)
    trim.push(hub)
    rotorDiscs.push(disc)
    rotorRings.push(ring)
  }

  return { group, rotors, body, arms: [arm1, arm2], trim, rotorDiscs, rotorRings, spin: 0 }
}

/**
 * Update one drone's transform + appearance for a frame.
 * - alive: bright/emissive, level, fast rotor spin.
 * - dead: dark, dropped slightly, tilted, slowly spinning down.
 */
export function updateDrone(
  d: Drone,
  x: number,
  y: number,
  z: number,
  yaw: number,
  alive: boolean,
  dt: number,
) {
  d.group.position.set(x, alive ? z : z - 0.9, y)
  d.group.rotation.y = yaw

  if (alive) {
    d.spin = 1
    d.group.rotation.z = 0
    d.group.rotation.x = 0
    d.group.scale.setScalar(1)
    if (d.body.material !== aliveBody) {
      d.body.material = aliveBody
      d.arms[0].material = aliveBody
      d.arms[1].material = aliveBody
      for (const mesh of d.trim) mesh.material = aliveTrim
      for (const disc of d.rotorDiscs) disc.material = aliveRotorDisc
      for (const ring of d.rotorRings) ring.material = aliveRotorRing
    }
    for (const r of d.rotors) r.rotation.y += dt * 44
  } else {
    // ease spin toward 0, list to one side
    d.spin = Math.max(0, d.spin - dt * 0.6)
    d.group.rotation.z = 0.35
    d.group.rotation.x = 0.15
    d.group.scale.setScalar(0.85)
    if (d.body.material !== deadBody) {
      d.body.material = deadBody
      d.arms[0].material = deadBody
      d.arms[1].material = deadBody
      for (const mesh of d.trim) mesh.material = deadRotor
      for (const disc of d.rotorDiscs) disc.material = deadRotorDisc
      for (const ring of d.rotorRings) ring.material = deadRotor
    }
    for (const r of d.rotors) r.rotation.y += dt * 44 * d.spin
  }
}

const TRAIL_LEN = 40

/** A fading line tracing a drone's recent positions. */
export class Trail {
  readonly line: THREE.Line
  private positions: Float32Array
  private head = 0
  private count = 0
  private geo: THREE.BufferGeometry

  constructor(color: number) {
    this.positions = new Float32Array(TRAIL_LEN * 3)
    this.geo = new THREE.BufferGeometry()
    this.geo.setAttribute(
      'position',
      new THREE.BufferAttribute(this.positions, 3),
    )
    const mat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.45,
    })
    this.line = new THREE.Line(this.geo, mat)
    this.line.frustumCulled = false
  }

  push(x: number, y: number, z: number) {
    const i = this.head * 3
    this.positions[i] = x
    this.positions[i + 1] = z
    this.positions[i + 2] = y
    this.head = (this.head + 1) % TRAIL_LEN
    if (this.count < TRAIL_LEN) this.count++
    // rebuild a contiguous oldest->newest range for drawing
    const ordered = this.geo.getAttribute('position') as THREE.BufferAttribute
    const arr = ordered.array as Float32Array
    for (let k = 0; k < this.count; k++) {
      const src = ((this.head - this.count + k + TRAIL_LEN) % TRAIL_LEN) * 3
      arr[k * 3] = this.positions[src]
      arr[k * 3 + 1] = this.positions[src + 1]
      arr[k * 3 + 2] = this.positions[src + 2]
    }
    this.geo.setDrawRange(0, this.count)
    ordered.needsUpdate = true
  }

  clear() {
    this.head = 0
    this.count = 0
    this.geo.setDrawRange(0, 0)
  }

  dispose() {
    this.geo.dispose()
    ;(this.line.material as THREE.Material).dispose()
  }
}
