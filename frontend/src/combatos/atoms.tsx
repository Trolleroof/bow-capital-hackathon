interface TrajPoint { x: number; y: number }

function fitPoints(points: TrajPoint[], w: number, h: number, pad: number) {
  const xs = points.map(p => p.x)
  const ys = points.map(p => p.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const sx = (w - 2 * pad) / ((maxX - minX) || 1)
  const sy = (h - 2 * pad) / ((maxY - minY) || 1)
  return points.map(p => [
    +(pad + (p.x - minX) * sx).toFixed(1),
    +(h - (pad + (p.y - minY) * sy)).toFixed(1),
  ])
}

export function TrajPlot({ points, w = 300, h = 160 }: { points: TrajPoint[]; w?: number; h?: number }) {
  if (points.length < 2) return null
  const fitted = fitPoints(points, w, h, 14)
  const d = fitted.map(p => p.join(',')).join(' ')
  const last = fitted[fitted.length - 1]
  return (
    <svg className="traj-svg" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <polyline className="traj-line" points={d} />
      <circle className="traj-dot" cx={last[0]} cy={last[1]} r="3.4" />
    </svg>
  )
}

type GaugeAccent = 'amber' | 'green' | 'red'

export function Gauge({ value, max = 100, accent = 'amber' }: { value: number; max?: number; accent?: GaugeAccent }) {
  return (
    <div className="gauge">
      <div className={`gauge-fill gauge-fill--${accent}`} style={{ width: Math.min(100, (value / max) * 100) + '%' }} />
    </div>
  )
}
