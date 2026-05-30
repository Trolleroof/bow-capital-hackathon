import { useState } from 'react'
import type { TelemetryState } from './useCombatState'
import { TrajPlot } from './atoms'

interface Props {
  t: TelemetryState
  onExit: () => void
  onConfirm: (id: string) => void
}

export function OpticView({ t, onExit, onConfirm }: Props) {
  const [locked, setLocked] = useState(false)

  const base = Math.round(t.heading / 15) * 15
  const headings = Array.from({ length: 5 }, (_, i) => ((base + (i - 2) * 15) + 360) % 360)

  const primary = t.dets.find(d => d.tone === 'amber')

  const handleLock = () => {
    if (locked) return
    setLocked(true)
    if (primary) onConfirm(primary.id)
  }

  return (
    <div className={'optic' + (locked ? ' is-locked' : '')}>
      <div className="feed-bg" />
      <div className="vign" />
      <div className="boot-sweep" />

      <div className="fcrn tl" /><div className="fcrn tr" />
      <div className="fcrn bl" /><div className="fcrn br" />

      <button className="exit-btn" onClick={onExit}>◄ EXIT OPTIC</button>
      <div className="opt-id">
        STEREO POD · 1280×720<br />
        ID <b>EUROC-07</b>
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

      {/* targeting reticle */}
      <div className="recticle-wrap">
        <div className="rt-corner tl" /><div className="rt-corner tr" />
        <div className="rt-corner bl" /><div className="rt-corner br" />
        <div className="rt-fill" />
        <div className="cross" />
        {!locked && <div className="rt-scan" />}
        {primary && (
          <div className="rt-tag">
            {locked ? `LOCKED · ${primary.id}` : `TRACKING · ${primary.id}`}
          </div>
        )}
        {primary && (
          <div className="rt-conf">{primary.conf.toFixed(2)}</div>
        )}
      </div>

      {/* lock / confirm button */}
      <button
        className={'lock-btn' + (locked ? ' is-locked' : '')}
        onClick={handleLock}
        disabled={locked}
      >
        {locked ? 'TARGET CONFIRMED' : 'CONFIRM TARGET'}
      </button>

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
