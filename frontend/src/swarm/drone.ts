/**
 * drone.ts — Three.js visuals for the swarm: a quadrotor-ish drone mesh and a
 * per-drone fading motion trail. Kept out of SwarmPanel so the React component
 * stays focused on the data/render loop.
 */
import * as THREE from 'three'

const ARM_LEN = 0.62
const ROTOR_R = 0.34

// Shared geometries/materials (created once, reused for every drone).
const bodyGeo = new THREE.BoxGeometry(0.55, 0.16, 0.55)
const armGeo = new THREE.BoxGeometry(ARM_LEN * 2, 0.06, 0.08)
const rotorGeo = new THREE.CylinderGeometry(ROTOR_R, ROTOR_R, 0.04, 16)

const aliveBody = new THREE.MeshStandardMaterial({
  color: 0x0e3a2a,
  emissive: 0x123b2a,
  emissiveIntensity: 0.6,
  metalness: 0.3,
  roughness: 0.5,
})
const aliveRotor = new THREE.MeshStandardMaterial({
  color: 0x4ef0a0,
  emissive: 0x2fae7a,
  emissiveIntensity: 0.9,
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

export interface Drone {
  group: THREE.Group
  rotors: THREE.Mesh[]
  body: THREE.Mesh
  arms: THREE.Mesh[]
  /** spin-down state for dead drones */
  spin: number
}

/** Build one reusable quadrotor: a body + two crossed arms + 4 rotor discs. */
export function makeDrone(): Drone {
  const group = new THREE.Group()

  const body = new THREE.Mesh(bodyGeo, aliveBody)
  group.add(body)

  const arm1 = new THREE.Mesh(armGeo, aliveBody)
  const arm2 = new THREE.Mesh(armGeo, aliveBody)
  arm2.rotation.y = Math.PI / 2
  group.add(arm1, arm2)

  const rotors: THREE.Mesh[] = []
  const offs: [number, number][] = [
    [ARM_LEN, ARM_LEN],
    [-ARM_LEN, ARM_LEN],
    [ARM_LEN, -ARM_LEN],
    [-ARM_LEN, -ARM_LEN],
  ]
  for (const [dx, dz] of offs) {
    const r = new THREE.Mesh(rotorGeo, aliveRotor)
    r.position.set(dx, 0.06, dz)
    group.add(r)
    rotors.push(r)
  }

  return { group, rotors, body, arms: [arm1, arm2], spin: 0 }
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
      for (const r of d.rotors) r.material = aliveRotor
    }
    for (const r of d.rotors) r.rotation.y += dt * 40
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
      for (const r of d.rotors) r.material = deadRotor
    }
    for (const r of d.rotors) r.rotation.y += dt * 40 * d.spin
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
