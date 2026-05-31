import { useEffect, useMemo, useState } from 'react'
import type { SlamPose } from './useCombatState'
import {
  formatLatLon,
  latLonToLocalPx,
  osmTileUrl,
  pathToLocalPx,
  slamToLatLon,
  tilesForViewport,
  type GeoAnchor,
} from './slamGeo'

interface Props {
  path: SlamPose[]
  pose: SlamPose | null
  size?: number
  zoom?: number
}

type GeoStatus = 'pending' | 'ready' | 'denied' | 'unsupported'

export function GeoMinimap({ path, pose, size = 150, zoom = 17 }: Props) {
  const [anchor, setAnchor] = useState<GeoAnchor | null>(null)
  const [geoStatus, setGeoStatus] = useState<GeoStatus>('pending')
  const [deviceLatLon, setDeviceLatLon] = useState<{ lat: number; lon: number } | null>(null)

  useEffect(() => {
    if (!navigator.geolocation) {
      setGeoStatus('unsupported')
      return undefined
    }

    const onSuccess = (position: GeolocationPosition) => {
      const { latitude: lat, longitude: lon } = position.coords
      setDeviceLatLon({ lat, lon })
      setGeoStatus('ready')
    }

    const onError = () => setGeoStatus('denied')

    navigator.geolocation.getCurrentPosition(onSuccess, onError, {
      enableHighAccuracy: true,
      timeout: 20_000,
      maximumAge: 5_000,
    })

    const watchId = navigator.geolocation.watchPosition(onSuccess, () => {}, {
      enableHighAccuracy: true,
      maximumAge: 10_000,
    })

    return () => navigator.geolocation.clearWatch(watchId)
  }, [])

  useEffect(() => {
    if (geoStatus !== 'ready' || !deviceLatLon || !pose || anchor) return
    setAnchor({
      lat: deviceLatLon.lat,
      lon: deviceLatLon.lon,
      slamX: pose.x,
      slamY: pose.y,
    })
  }, [geoStatus, deviceLatLon, pose, anchor])

  const center = useMemo(() => {
    if (anchor && pose) return slamToLatLon(pose.x, pose.y, anchor)
    if (deviceLatLon) return deviceLatLon
    return null
  }, [anchor, pose, deviceLatLon])

  const tiles = useMemo(
    () => (center ? tilesForViewport(center.lat, center.lon, zoom, size) : []),
    [center, zoom, size],
  )

  const pathPx = useMemo(() => {
    if (!anchor || !center) return []
    return pathToLocalPx(path.slice(-80), anchor, center.lat, center.lon, zoom, size)
  }, [anchor, center, path, zoom, size])

  const posePx = useMemo(() => {
    if (!anchor || !center || !pose) return null
    const { lat, lon } = slamToLatLon(pose.x, pose.y, anchor)
    return latLonToLocalPx(lat, lon, center.lat, center.lon, zoom, size)
  }, [anchor, center, pose, zoom, size])

  const pathD = pathPx
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`)
    .join(' ')

  const coordLabel = (() => {
    if (geoStatus === 'pending') return 'ACQUIRING GPS…'
    if (geoStatus === 'unsupported') return 'NO GEO API'
    if (geoStatus === 'denied') return 'GPS DENIED'
    if (anchor && pose) {
      const { lat, lon } = slamToLatLon(pose.x, pose.y, anchor)
      return formatLatLon(lat, lon)
    }
    if (deviceLatLon) return formatLatLon(deviceLatLon.lat, deviceLatLon.lon)
    return 'NO FIX'
  })()

  return (
    <div className="vslam-pip" aria-label="OpenStreetMap location from device GPS and SLAM offset">
      <div className="vslam-pip-head">
        <span>OSM · GPS+SLAM</span>
        <b title={coordLabel}>{coordLabel}</b>
      </div>
      <div className="vslam-map" style={{ width: size, height: size }}>
        {center ? (
          <div className="vslam-map-tiles" style={{ width: size, height: size }}>
            {tiles.map(tile => (
              <img
                key={tile.key}
                className="vslam-map-tile"
                src={osmTileUrl(zoom, tile.x, tile.y)}
                alt=""
                draggable={false}
                style={{
                  left: tile.left,
                  top: tile.top,
                  width: 256,
                  height: 256,
                }}
              />
            ))}
          </div>
        ) : (
          <div className="vslam-map-wait">AWAITING GPS FIX…</div>
        )}
        <svg
          viewBox={`0 0 ${size} ${size}`}
          className="vslam-map-overlay"
          role="img"
          aria-hidden={!center}
        >
          {pathD && <path className="map-slam-path" d={pathD} />}
          {posePx && (
            <g className="map-pose" transform={`translate(${posePx.x.toFixed(1)} ${posePx.y.toFixed(1)})`}>
              <circle r="6" />
              <path d="M0 -10 L3 -2 L0 0 L-3 -2 Z" />
            </g>
          )}
        </svg>
        <div className="vslam-map-attrib">© OpenStreetMap</div>
      </div>
    </div>
  )
}
