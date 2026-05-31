import { useEffect, useMemo, useState } from 'react'
import { LiveFrameCanvas } from './LiveFrameCanvas'
import type { Detection, TelemetryState } from './useOutcastVirusState'

interface Props {
  t: TelemetryState
  onExit: () => void
  onFollow: (numericId: number, id: string) => void
  onConfirm: (numericId: number, id: string) => void
  onRelease: () => void
}

const COLOR: Record<string, string> = {
  confirmed: '#50ff00',
  primary: '#ff3c3c',
  candidate: '#ff50c8',
  troop: '#ffdc00',
  vehicle: '#ffa000',
  ugv: '#78c81e',
  aerial: '#32f0ff',
  mute: '#404040',
  default: '#8c8c8c',
}

function detColor(d: Detection): string {
  if (d.confirmed) return COLOR.confirmed
  if (d.tone === 'amber') return COLOR.primary
  if (d.tone === 'candidate') return COLOR.candidate
  if (d.tone === 'mute') return COLOR.mute
  return COLOR[d.cls.toLowerCase()] ?? COLOR.default
}

interface DetectionOverlayOptions {
  lineScale?: number
}

export function makeDetectionOverlay(displayDets: Detection[], options: DetectionOverlayOptions = {}) {
  return (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const lineScale = options.lineScale ?? 1
    for (const d of displayDets) {
      if (!d.bbox) continue
      const [nx, ny, nw, nh] = d.bbox
      const x = nx * width
      const y = ny * height
      const w = nw * width
      const h = nh * height
      const color = detColor(d)
      const thick = (d.confirmed || d.tone === 'amber' ? 8 : 6) * lineScale
      const frac = d.confirmed ? 0.3 : 0.24

      drawCorners(ctx, x, y, w, h, color, thick, frac)

      if (d.confirmed || d.tone === 'amber') {
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(x + w / 2, y + h / 2, 4.5, 0, Math.PI * 2)
        ctx.fill()
      }

      const state = d.confirmed ? 'CONFIRMED' : d.tone === 'amber' ? 'LOCKED' : d.tone === 'candidate' ? 'FOLLOW' : ''
      const iffTag = d.allegiance ? `  ${d.allegiance.toUpperCase()}` : ''
      const label = `${d.id}  ${d.cls}  ${d.conf.toFixed(2)}${state ? '  ' + state : ''}${iffTag}`
      drawLabel(ctx, label, x + w / 2, y - 4, color, d.confirmed)
    }
  }
}

function drawCorners(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  thick = 3,
  frac = 0.24,
) {
  const lx = Math.max(8, w * frac)
  const ly = Math.max(8, h * frac)
  ctx.strokeStyle = color
  ctx.lineWidth = thick
  ctx.lineCap = 'square'
  ctx.lineJoin = 'miter'
  for (const [ox, oy, sx, sy] of [
    [x, y, 1, 1],
    [x + w, y, -1, 1],
    [x, y + h, 1, -1],
    [x + w, y + h, -1, -1],
  ] as [number, number, number, number][]) {
    ctx.beginPath()
    ctx.moveTo(ox, oy)
    ctx.lineTo(ox + sx * lx, oy)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(ox, oy)
    ctx.lineTo(ox, oy + sy * ly)
    ctx.stroke()
  }
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  cx: number,
  top: number,
  color: string,
  bold = false,
) {
  const fs = bold ? 20 : 18
  ctx.font = `${bold ? 700 : 400} ${fs}px monospace`
  const tw = ctx.measureText(text).width
  const pad = 4
  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  ctx.fillRect(cx - tw / 2 - pad, top - fs - pad, tw + pad * 2, fs + pad * 2)
  ctx.fillStyle = color
  ctx.fillText(text, cx - tw / 2, top)
}

export function OpticView({ t, onExit, onFollow, onConfirm, onRelease }: Props) {
  const [followedId, setFollowedId] = useState<number | null>(null)
  const [confirmedId, setConfirmedId] = useState<number | null>(null)

  const displayDets = useMemo(() => t.dets.map(d => ({
    ...d,
    tone: (confirmedId === d.numericId ? ''
      : followedId === d.numericId ? 'amber'
      : d.tone) as Detection['tone'],
    confirmed: d.confirmed || confirmedId === d.numericId,
  })), [t.dets, followedId, confirmedId])

  const candidate = displayDets.find(d => d.tone === 'candidate')
    ?? displayDets.find(d => d.st !== 'LOST' && d.numericId !== followedId && d.numericId !== confirmedId)
  const followedDet = followedId != null ? displayDets.find(d => d.numericId === followedId) : null
  const confirmedDet = confirmedId != null ? displayDets.find(d => d.numericId === confirmedId) : null
  const primary = confirmedDet ?? followedDet
  const liveTargets = displayDets.filter(d => d.st !== 'LOST')
  const tapeTarget = primary ?? candidate ?? displayDets.find(d => d.st !== 'LOST') ?? null
  const tapeState = tapeTarget
    ? tapeTarget.confirmed
      ? 'CONF'
      : tapeTarget.tone === 'amber'
        ? 'TRACK'
        : tapeTarget.tone === 'candidate'
          ? 'NEXT'
          : tapeTarget.st
    : '--'
  const classCounts = liveTargets.reduce<Record<string, number>>((acc, det) => {
    acc[det.cls] = (acc[det.cls] ?? 0) + 1
    return acc
  }, {})
  const topClasses = Object.entries(classCounts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 3)

  const overlay = useMemo(() => makeDetectionOverlay(displayDets), [displayDets])

  const handleFollow = () => {
    if (!candidate) return
    setFollowedId(candidate.numericId)
    setConfirmedId(null)
    onFollow(candidate.numericId, candidate.id)
  }

  const handleConfirm = () => {
    if (!followedDet) return
    setConfirmedId(followedDet.numericId)
    onConfirm(followedDet.numericId, followedDet.id)
  }

  const handleRelease = () => {
    if (confirmedId != null) {
      setConfirmedId(null)
    } else {
      setFollowedId(null)
    }
    onRelease()
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === 'f' || e.key === 'F') { e.preventDefault(); handleFollow() }
      else if (e.key === 'c' || e.key === 'C') { e.preventDefault(); handleConfirm() }
      else if (e.key === 'r' || e.key === 'R') { e.preventDefault(); handleRelease() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <div className={'optic' + (confirmedId != null ? ' is-locked' : '')}>
      {t.cameraFrame
        ? <LiveFrameCanvas frame={t.cameraFrame} className="feed-bg feed-canvas" fit="cover" overlay={overlay} />
        : <div className="feed-bg" />}

      <div className="vign" />
      <div className="boot-sweep" />

      <div className="fcrn tl" /><div className="fcrn tr" />
      <div className="fcrn bl" /><div className="fcrn br" />

      <button className="exit-btn" onClick={onExit}>EXIT OPTIC</button>

      <div className="tape l">
        <div className="tl">TARGET</div>
        <div className="tr">ID<b>{tapeTarget?.id ?? '--'}</b></div>
        <div className="tr">CLS<b>{tapeTarget?.cls ?? '--'}</b></div>
        <div className="tr">CONF<b>{tapeTarget ? tapeTarget.conf.toFixed(2) : '--'}</b></div>
        <div className="tr">STATE<b>{tapeState}</b></div>
      </div>
      <div className="tape r">
        <div className="tl">TARGETS</div>
        <div className="tr">TOTAL<b>{liveTargets.length}</b></div>
        {topClasses.length > 0 ? (
          topClasses.map(([cls, count]) => (
            <div className="tr" key={cls}>
              {cls}<b>{count}</b>
            </div>
          ))
        ) : (
          <>
            <div className="tr">TYPE 1<b>--</b></div>
            <div className="tr">TYPE 2<b>--</b></div>
            <div className="tr">TYPE 3<b>--</b></div>
          </>
        )}
      </div>


<div className="kbd-legend">
        <div className="kbd-row"><span className="kbd-key">F</span><span className="kbd-desc">FOLLOW TARGET</span></div>
        <div className="kbd-row"><span className="kbd-key">C</span><span className="kbd-desc">CONFIRM TARGET</span></div>
        <div className="kbd-row"><span className="kbd-key">R</span><span className="kbd-desc">RELEASE</span></div>
      </div>
    </div>
  )
}
