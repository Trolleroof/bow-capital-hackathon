import { useEffect, useMemo, useRef, useState } from 'react'
import type { TelemetryState, Detection } from './useCombatState'
import { TrajPlot } from './atoms'

interface Props {
  t: TelemetryState
  onExit: () => void
  onFollow: (numericId: number, id: string) => void
  onConfirm: (numericId: number, id: string) => void
  onRelease: () => void
}

// ── palette (matches perception/visualizer.py BGR→hex exactly) ───────────────
const COLOR: Record<string, string> = {
  confirmed:  '#50ff00',  // (0,255,80) BGR
  primary:    '#ff3c3c',  // (60,60,255) BGR  -- followed
  candidate:  '#ff50c8',  // (200,80,255) BGR -- proposed / buffer pick
  troop:      '#ffdc00',  // (0,220,255) BGR
  vehicle:    '#ffa000',  // (0,160,255) BGR
  ugv:        '#78c81e',  // (30,200,120) BGR
  aerial:     '#32f0ff',  // (255,240,50) BGR
  mute:       '#404040',
  default:    '#8c8c8c',
}

function detColor(d: Detection): string {
  if (d.confirmed)             return COLOR.confirmed
  if (d.tone === 'amber')      return COLOR.primary
  if (d.tone === 'candidate')  return COLOR.candidate
  if (d.tone === 'mute')       return COLOR.mute
  return COLOR[d.cls.toLowerCase()] ?? COLOR.default
}

// ── canvas drawing helpers ────────────────────────────────────────────────────
function drawCorners(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
  color: string, thick = 1, frac = 0.22,
) {
  const lx = Math.max(6, w * frac)
  const ly = Math.max(6, h * frac)
  ctx.strokeStyle = color
  ctx.lineWidth   = thick
  ctx.lineCap     = 'square'
  for (const [ox, oy, sx, sy] of [
    [x,     y,     1,  1],
    [x + w, y,    -1,  1],
    [x,     y + h, 1, -1],
    [x + w, y + h,-1, -1],
  ] as [number, number, number, number][]) {
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + sx * lx, oy);          ctx.stroke()
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox,           oy + sy * ly); ctx.stroke()
  }
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string, cx: number, top: number, color: string, bold = false,
) {
  const fs   = bold ? 11 : 10
  ctx.font   = `${bold ? 700 : 400} ${fs}px monospace`
  const tw   = ctx.measureText(text).width
  const pad  = 3
  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  ctx.fillRect(cx - tw / 2 - pad, top - fs - pad, tw + pad * 2, fs + pad * 2)
  ctx.fillStyle = color
  ctx.fillText(text, cx - tw / 2, top)
}

// ── overlay canvas ────────────────────────────────────────────────────────────
function DetectionOverlay({ dets }: { dets: Detection[] }) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)

    for (const d of dets) {
      if (!d.bbox) continue
      const [nx, ny, nw, nh] = d.bbox
      const x = nx * W, y = ny * H, w = nw * W, h = nh * H
      const color = detColor(d)
      const thick = d.confirmed || d.tone === 'amber' ? 2 : 1
      const frac  = d.confirmed ? 0.28 : 0.22

      drawCorners(ctx, x, y, w, h, color, thick, frac)

      if (d.confirmed || d.tone === 'amber') {
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(x + w / 2, y + h / 2, 3, 0, Math.PI * 2)
        ctx.fill()
      }

      const state = d.confirmed ? 'CONFIRMED' : d.tone === 'amber' ? 'LOCKED' : d.tone === 'candidate' ? '→FOLLOW' : ''
      const label = `${d.id}  ${d.cls}  ${d.conf.toFixed(2)}${state ? '  ' + state : ''}`
      drawLabel(ctx, label, x + w / 2, y - 4, color, d.confirmed)
    }
  }, [dets])

  return (
    <canvas
      ref={ref}
      width={1440}
      height={810}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    />
  )
}

// ── main view ────────────────────────────────────────────────────────────────
export function OpticView({ t, onExit, onFollow, onConfirm, onRelease }: Props) {
  const base     = Math.round(t.heading / 15) * 15
  const headings = Array.from({ length: 5 }, (_, i) => ((base + (i - 2) * 15) + 360) % 360)

  // Local operator state -- decoupled from WS tone so FOLLOW must be pressed first
  const [followedId, setFollowedId]   = useState<number | null>(null)
  const [confirmedId, setConfirmedId] = useState<number | null>(null)

  // Local operator state always wins over WS tone to prevent bus-message flicker.
  // When the operator follows/confirms a track, apply that state on top of whatever
  // the backend last sent -- the backend will catch up within one round-trip.
  const displayDets = useMemo(() => t.dets.map(d => ({
    ...d,
    tone: (confirmedId === d.numericId ? ''
           : followedId === d.numericId ? 'amber'
           : d.tone) as Detection['tone'],
    confirmed: d.confirmed || confirmedId === d.numericId,
  })), [t.dets, followedId, confirmedId])

  // Candidate is whatever the perception buffer proposed (magenta); fall back to any track
  const candidate = displayDets.find(d => d.tone === 'candidate')
    ?? displayDets.find(d => d.st !== 'LOST' && d.numericId !== followedId && d.numericId !== confirmedId)
  const followedDet  = followedId  != null ? displayDets.find(d => d.numericId === followedId)  : null
  const confirmedDet = confirmedId != null ? displayDets.find(d => d.numericId === confirmedId) : null
  const primary = confirmedDet ?? followedDet

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
      setConfirmedId(null)   // confirmed → followed
    } else {
      setFollowedId(null)    // followed → none
    }
    onRelease()
  }

  return (
    <div className={'optic' + (confirmedId != null ? ' is-locked' : '')}>
      {/* raw video background */}
      {t.cameraFrame
        ? <img className="feed-bg" src={t.cameraFrame.data} alt="" draggable={false} />
        : <div className="feed-bg" />
      }

      {/* frontend detection overlay */}
      <DetectionOverlay dets={displayDets} />

      <div className="vign" />
      <div className="boot-sweep" />

      <div className="fcrn tl" /><div className="fcrn tr" />
      <div className="fcrn bl" /><div className="fcrn br" />

      <button className="exit-btn" onClick={onExit}>◄ EXIT OPTIC</button>
      <div className="opt-id">
        STEREO POD · {t.cameraFrame ? `${t.cameraFrame.width || 640}×${t.cameraFrame.height || 360}` : '---'}<br />
        ID <b>EDGE-NODE</b>
      </div>

      <div className="top-status">
        <span className="s deny"><span className="k">GPS </span><b>DENIED</b></span>
        <span className="sep">·</span>
        <span className="s deny"><span className="k">LINK </span><b>NONE</b></span>
        <span className="sep">·</span>
        <span className="s"><b>{t.tracking === 'OK' ? 'LOCALIZED' : t.tracking}</b></span>
      </div>

      <div className="heading">
        {headings.map((h, i) => (
          <div className="htk" key={i}>{String(h).padStart(3, '0')}</div>
        ))}
        <div className="hctr">▼</div>
      </div>

      <div className="tape l">
        <div className="tl">POSE · M</div>
        <div className="tr">X<b>{t.pose.x >= 0 ? '+' : ''}{t.pose.x.toFixed(1)}</b></div>
        <div className="tr">Y<b>{t.pose.y.toFixed(1)}</b></div>
        <div className="tr">Z<b>{t.pose.z >= 0 ? '+' : ''}{t.pose.z.toFixed(1)}</b></div>
        <div className="tr">V<b>{t.vel.toFixed(2)}</b></div>
      </div>
      <div className="tape r">
        <div className="tl">EDGE</div>
        <div className="tr">SLAM<b>{t.slam}</b></div>
        <div className="tr">YOLO<b>{t.yolo}</b></div>
        <div className="tr">GPU<b>{t.gpu}%</b></div>
        <div className="tr">DRIFT<b>{t.drift.toFixed(1)}</b></div>
      </div>

      {/* primary target reticle */}
      {primary && (
        <div className="recticle-wrap">
          <div className="rt-corner tl" /><div className="rt-corner tr" />
          <div className="rt-corner bl" /><div className="rt-corner br" />
          <div className="rt-fill" />
          <div className="cross" />
          {confirmedId == null && <div className="rt-scan" />}
          <div className="rt-tag">{confirmedId != null ? `CONFIRMED · ${primary.id}` : `TRACKING · ${primary.id}`}</div>
          <div className="rt-conf">{primary.conf.toFixed(2)}</div>
        </div>
      )}

      {/* ── operator button strip ── */}
      <div className="op-strip">
        {followedId == null && confirmedId == null && (
          <button className="op-btn op-btn--follow" onClick={handleFollow} disabled={!candidate}>
            FOLLOW {candidate ? candidate.id : 'TARGET'}
          </button>
        )}
        {followedId != null && confirmedId == null && (
          <>
            <button className="op-btn op-btn--confirm" onClick={handleConfirm}>
              CONFIRM TARGET
            </button>
            <button className="op-btn op-btn--release" onClick={handleRelease}>
              RELEASE
            </button>
          </>
        )}
        {confirmedId != null && (
          <button className="op-btn op-btn--release" onClick={handleRelease}>
            RELEASE LOCK
          </button>
        )}
      </div>

      <div className="cread bl">
        <div className="ct">NAV / VSLAM</div>
        TRACK <b>{t.tracking}</b><br />
        LOOPS <b>{t.loops}</b><br />
        MODE <b>STEREO</b>
      </div>
      <div className="cread br">
        <div className="ct">RECON / 3DGS</div>
        STATUS <b>{t.recon.status === 'ready' ? 'READY' : 'TRAINING'}</b><br />
        FRAMES <b>{t.recon.frames || 220}</b><br />
        FLY-THRU <b>{t.recon.status === 'ready' ? 'ARMED' : 'PENDING'}</b>
      </div>

      <div className="bottom-strip">
        <div className="ribbon">
          <div className="rl">TRAJECTORY · LIVE</div>
          <TrajPlot points={t.traj} w={380} h={38} />
        </div>
      </div>
    </div>
  )
}
