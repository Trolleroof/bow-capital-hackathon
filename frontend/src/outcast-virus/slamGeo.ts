import type { SlamPose } from './useOutcastVirusState'

/** Meters per degree latitude (WGS84 approximation). */
const M_PER_DEG_LAT = 111_320

export interface GeoAnchor {
  lat: number
  lon: number
  slamX: number
  slamY: number
}

export function metersPerDegLon(lat: number) {
  return M_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180)
}

/** SLAM frame (x east, y north in meters) → WGS84 lat/lon from a paired GPS+SLAM anchor. */
export function slamToLatLon(slamX: number, slamY: number, anchor: GeoAnchor) {
  const dEast = slamX - anchor.slamX
  const dNorth = slamY - anchor.slamY
  return {
    lat: anchor.lat + dNorth / M_PER_DEG_LAT,
    lon: anchor.lon + dEast / metersPerDegLon(anchor.lat),
  }
}

export function formatLatLon(lat: number, lon: number) {
  const latHem = lat >= 0 ? 'N' : 'S'
  const lonHem = lon >= 0 ? 'E' : 'W'
  return `${Math.abs(lat).toFixed(5)}°${latHem} ${Math.abs(lon).toFixed(5)}°${lonHem}`
}

/** Web Mercator world pixel at zoom z for a lat/lon point. */
export function latLonToWorldPx(lat: number, lon: number, zoom: number) {
  const scale = 256 * 2 ** zoom
  const x = ((lon + 180) / 360) * scale
  const sinLat = Math.sin((lat * Math.PI) / 180)
  const y = (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale
  return { x, y }
}

export function latLonToLocalPx(
  lat: number,
  lon: number,
  centerLat: number,
  centerLon: number,
  zoom: number,
  size: number,
) {
  const point = latLonToWorldPx(lat, lon, zoom)
  const center = latLonToWorldPx(centerLat, centerLon, zoom)
  return {
    x: size / 2 + (point.x - center.x),
    y: size / 2 + (point.y - center.y),
  }
}

export interface MapTile {
  key: string
  x: number
  y: number
  left: number
  top: number
}

/** Tiles needed to cover a square viewport centered on lat/lon. */
export function tilesForViewport(
  centerLat: number,
  centerLon: number,
  zoom: number,
  size: number,
): MapTile[] {
  const center = latLonToWorldPx(centerLat, centerLon, zoom)
  const half = size / 2
  const minX = center.x - half
  const maxX = center.x + half
  const minY = center.y - half
  const maxY = center.y + half

  const tileMinX = Math.floor(minX / 256)
  const tileMaxX = Math.floor(maxX / 256)
  const tileMinY = Math.floor(minY / 256)
  const tileMaxY = Math.floor(maxY / 256)

  const tiles: MapTile[] = []
  for (let tx = tileMinX; tx <= tileMaxX; tx++) {
    for (let ty = tileMinY; ty <= tileMaxY; ty++) {
      tiles.push({
        key: `${zoom}/${tx}/${ty}`,
        x: tx,
        y: ty,
        left: tx * 256 - center.x + half,
        top: ty * 256 - center.y + half,
      })
    }
  }
  return tiles
}

const CARTO_SUBDOMAINS = ['a', 'b', 'c', 'd'] as const

/** CARTO Dark Matter (no labels) — same provider as worldhacks.xyz, dark variant for Outcast Virus. */
export function mapTileUrl(z: number, x: number, y: number, retina = false) {
  const sub = CARTO_SUBDOMAINS[(x + y) % CARTO_SUBDOMAINS.length]
  const suffix = retina ? '@2x' : ''
  return `https://${sub}.basemaps.cartocdn.com/dark_nolabels/${z}/${x}/${y}${suffix}.png`
}

export const MAP_ATTRIBUTION = '© OpenStreetMap © CARTO'

export function pathToLocalPx(
  path: SlamPose[],
  anchor: GeoAnchor,
  centerLat: number,
  centerLon: number,
  zoom: number,
  size: number,
) {
  return path.map(point => {
    const { lat, lon } = slamToLatLon(point.x, point.y, anchor)
    return latLonToLocalPx(lat, lon, centerLat, centerLon, zoom, size)
  })
}
