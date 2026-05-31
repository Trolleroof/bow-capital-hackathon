import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { SlamPoint, SlamPose } from './useCombatState'

interface Props {
  points: SlamPoint[]
  path: SlamPose[]
  pose: SlamPose | null
}

function rosToThree(point: SlamPoint) {
  return new THREE.Vector3(point.x, point.z, -point.y)
}

function boundsFor(points: SlamPoint[], path: SlamPose[], pose: SlamPose | null) {
  const all = [...points, ...path]
  if (pose) all.push(pose)
  if (all.length === 0) return { center: new THREE.Vector3(), radius: 8 }

  const box = new THREE.Box3()
  for (const point of all) box.expandByPoint(rosToThree(point))
  const center = new THREE.Vector3()
  box.getCenter(center)
  const radius = Math.max(2, box.getSize(new THREE.Vector3()).length() * 0.56)
  return { center, radius }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function mapPoint(point: SlamPoint, origin: SlamPoint, scale: number, size: number) {
  return {
    x: clamp(size / 2 + (point.x - origin.x) * scale, 8, size - 8),
    y: clamp(size / 2 - (point.y - origin.y) * scale, 8, size - 8),
  }
}

export function VslamScene({ points, path, pose }: Props) {
  const [chaseEnabled, setChaseEnabled] = useState(true)
  const hostRef = useRef<HTMLDivElement>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const pointsRef = useRef<THREE.Points | null>(null)
  const pathRef = useRef<THREE.Line | null>(null)
  const poseRef = useRef<THREE.Group | null>(null)
  const frameRef = useRef<number | null>(null)
  const hasFramedRef = useRef(false)
  const chaseEnabledRef = useRef(true)
  const mapSize = 150
  const mapOrigin = pose ?? path[path.length - 1] ?? { x: 0, y: 0, z: 0 }
  const mapScale = 4.2
  const mapPose = pose ? mapPoint(pose, mapOrigin, mapScale, mapSize) : null
  const mapPath = path.slice(-80).map(point => mapPoint(point, mapOrigin, mapScale, mapSize))
  const mapPathD = mapPath.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(' ')

  useEffect(() => {
    chaseEnabledRef.current = chaseEnabled
    const controls = controlsRef.current
    if (controls) controls.enabled = !chaseEnabled
  }, [chaseEnabled])

  useEffect(() => {
    const host = hostRef.current
    if (!host) return undefined

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x05080d)

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 2000)
    camera.position.set(9, 7, 11)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    host.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.rotateSpeed = 0.65
    controls.panSpeed = 0.75
    controls.zoomSpeed = 0.8
    controls.minDistance = 0.4
    controls.maxDistance = 220
    controls.target.set(0, 0, 0)
    controls.enabled = !chaseEnabledRef.current

    const grid = new THREE.GridHelper(24, 24, 0x2f5d82, 0x162638)
    grid.position.y = -0.02
    scene.add(grid)

    const axes = new THREE.AxesHelper(1.4)
    scene.add(axes)

    const pointGeometry = new THREE.BufferGeometry()
    const pointMaterial = new THREE.PointsMaterial({ color: 0x74c8ff, size: 0.055, sizeAttenuation: true })
    const pointCloud = new THREE.Points(pointGeometry, pointMaterial)
    scene.add(pointCloud)

    const pathGeometry = new THREE.BufferGeometry()
    const pathMaterial = new THREE.LineBasicMaterial({ color: 0x37b8a4 })
    const pathLine = new THREE.Line(pathGeometry, pathMaterial)
    scene.add(pathLine)

    const poseGroup = new THREE.Group()
    const body = new THREE.Mesh(
      new THREE.ConeGeometry(0.22, 0.52, 4),
      new THREE.MeshBasicMaterial({ color: 0xe76a5b }),
    )
    body.rotation.x = Math.PI / 2
    poseGroup.add(body)
    const mast = new THREE.Mesh(
      new THREE.SphereGeometry(0.09, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    )
    mast.position.y = 0.24
    poseGroup.add(mast)
    scene.add(poseGroup)

    cameraRef.current = camera
    controlsRef.current = controls
    pointsRef.current = pointCloud
    pathRef.current = pathLine
    poseRef.current = poseGroup

    const resize = () => {
      const width = Math.max(1, host.clientWidth)
      const height = Math.max(1, host.clientHeight)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }

    const render = () => {
      if (chaseEnabledRef.current && poseGroup.visible) {
        const chaseOffset = new THREE.Vector3(0, 1.2, 3.4).applyQuaternion(poseGroup.quaternion)
        const lookOffset = new THREE.Vector3(0, 0.18, -1.0).applyQuaternion(poseGroup.quaternion)
        const targetPosition = poseGroup.position.clone().add(chaseOffset)
        const targetLook = poseGroup.position.clone().add(lookOffset)
        camera.position.lerp(targetPosition, 0.18)
        controls.target.lerp(targetLook, 0.22)
      }
      controls.update()
      renderer.render(scene, camera)
      frameRef.current = requestAnimationFrame(render)
    }

    resize()
    render()
    window.addEventListener('resize', resize)

    return () => {
      window.removeEventListener('resize', resize)
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
      host.removeChild(renderer.domElement)
      pointGeometry.dispose()
      pointMaterial.dispose()
      pathGeometry.dispose()
      pathMaterial.dispose()
      controls.dispose()
      body.geometry.dispose()
      ;(body.material as THREE.Material).dispose()
      mast.geometry.dispose()
      ;(mast.material as THREE.Material).dispose()
      renderer.dispose()
    }
  }, [])

  useEffect(() => {
    const pointCloud = pointsRef.current
    const pathLine = pathRef.current
    const poseGroup = poseRef.current
    const camera = cameraRef.current
    if (!pointCloud || !pathLine || !poseGroup || !camera) return

    const pointPositions = new Float32Array(points.length * 3)
    points.forEach((point, index) => {
      const converted = rosToThree(point)
      pointPositions[index * 3] = converted.x
      pointPositions[index * 3 + 1] = converted.y
      pointPositions[index * 3 + 2] = converted.z
    })
    pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(pointPositions, 3))
    pointCloud.geometry.computeBoundingSphere()

    const pathPositions = new Float32Array(path.length * 3)
    path.forEach((point, index) => {
      const converted = rosToThree(point)
      pathPositions[index * 3] = converted.x
      pathPositions[index * 3 + 1] = converted.y
      pathPositions[index * 3 + 2] = converted.z
    })
    pathLine.geometry.setAttribute('position', new THREE.BufferAttribute(pathPositions, 3))
    pathLine.geometry.computeBoundingSphere()

    poseGroup.visible = Boolean(pose)
    if (pose) {
      poseGroup.position.copy(rosToThree(pose))
      if (typeof pose.qw === 'number') {
        poseGroup.quaternion.set(pose.qx ?? 0, pose.qz ?? 0, -(pose.qy ?? 0), pose.qw)
      }
    }

    const { center, radius } = boundsFor(points, path, pose)
    const controls = controlsRef.current
    const shouldFrame = !hasFramedRef.current && (points.length > 0 || path.length > 0 || pose)
    if (shouldFrame) {
      camera.position.set(center.x + radius * 0.85, center.y + radius * 0.65, center.z + radius * 1.05)
      controls?.target.copy(center)
      camera.lookAt(center)
      hasFramedRef.current = true
    }
    camera.near = Math.max(0.01, radius / 1000)
    camera.far = Math.max(50, radius * 8)
    camera.updateProjectionMatrix()
  }, [points, path, pose])

  return (
    <div className="vslam-wrap">
      <div ref={hostRef} className="vslam-scene" />
      <button
        type="button"
        className={`vslam-chase${chaseEnabled ? ' is-on' : ''}`}
        onClick={() => setChaseEnabled(value => !value)}
        aria-pressed={chaseEnabled}
        title={chaseEnabled ? 'Return to manual orbit camera' : 'Follow current SLAM camera pose'}
      >
        {chaseEnabled ? 'CHASE ON' : 'CHASE'}
      </button>
      <div className="vslam-pip" aria-label="Spoofed OpenStreetMap location inset">
        <div className="vslam-pip-head">
          <span>OSM SIM</span>
          <b>{pose ? `${pose.x.toFixed(1)}, ${pose.y.toFixed(1)}` : 'NO POSE'}</b>
        </div>
        <svg viewBox={`0 0 ${mapSize} ${mapSize}`} className="vslam-map" role="img">
          <rect className="map-land" x="0" y="0" width={mapSize} height={mapSize} />
          <path className="map-park" d="M9 98 C30 83 45 93 62 76 C83 55 111 66 140 46 L150 72 C132 88 119 118 92 130 C57 146 31 128 9 139 Z" />
          <path className="map-water" d="M0 35 C24 25 39 31 57 20 C82 5 106 9 150 0 L150 18 C119 20 92 16 70 32 C49 47 25 43 0 55 Z" />
          <g className="map-minor">
            <path d="M-8 28 L158 118" />
            <path d="M16 -8 L130 158" />
            <path d="M-5 125 L153 43" />
            <path d="M80 -10 L43 160" />
            <path d="M126 -8 L18 158" />
          </g>
          <g className="map-major">
            <path d="M-10 78 C31 71 50 88 86 72 C113 60 132 42 160 39" />
            <path d="M34 -8 C31 31 48 58 39 89 C32 113 36 132 52 158" />
            <path d="M-6 12 C36 42 71 50 104 82 C125 102 137 128 157 143" />
          </g>
          <g className="map-blocks">
            <rect x="19" y="54" width="17" height="10" />
            <rect x="43" y="35" width="21" height="13" />
            <rect x="81" y="29" width="18" height="12" />
            <rect x="105" y="91" width="22" height="14" />
            <rect x="62" y="112" width="25" height="13" />
            <rect x="19" y="110" width="20" height="12" />
          </g>
          {mapPathD && <path className="map-slam-path" d={mapPathD} />}
          {mapPose && (
            <g className="map-pose" transform={`translate(${mapPose.x.toFixed(1)} ${mapPose.y.toFixed(1)})`}>
              <circle r="6" />
              <path d="M0 -10 L3 -2 L0 0 L-3 -2 Z" />
            </g>
          )}
        </svg>
      </div>
    </div>
  )
}
