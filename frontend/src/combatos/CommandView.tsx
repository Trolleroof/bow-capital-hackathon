import { useMemo, useState } from 'react'
import { LiveFrameCanvas } from './LiveFrameCanvas'
import type { TelemetryState, LogEntry } from './useCombatState'
import { Gauge } from './atoms'
import { VslamScene } from './VslamScene'
import { makeDetectionOverlay } from './OpticView'

const MODULES = ['NAVIGATION', 'TARGETS', 'RECON', 'SYSTEM']

interface Props {
  t: TelemetryState
  log: LogEntry[]
  onEnterOptic: () => void
  onConfirm: (numericId: number, id: string) => void
}

export function CommandView({ t, log, onEnterOptic, onConfirm }: Props) {
  const [fullscreenPanel, setFullscreenPanel] = useState<'slam-map' | 'slam-keyframe' | null>(null)
  const clock = 'T+' + String(Math.floor(t.sec / 60)).padStart(2, '0') + ':' + String(t.sec % 60).padStart(2, '0')
  const tracked = t.dets.filter(d => d.st !== 'LOST').length
  const liveFeed = t.cameraFrame
  const annotatedFeed = t.slamFrame
  const liveFeedAspect = liveFeed && liveFeed.width > 0 && liveFeed.height > 0
    ? `${liveFeed.width} / ${liveFeed.height}`
    : '16 / 9'
  const cameraOverlay = useMemo(() => makeDetectionOverlay(t.dets, { lineScale: 0.42 }), [t.dets])
  const vslamPose = t.slamOdometry ?? { ...t.pose, qz: Math.sin(t.yaw * Math.PI / 360), qw: Math.cos(t.yaw * Math.PI / 360) }

  return (
    <div className="cmd">
      {/* hero bar */}
      <div className="hero">
        <div className="brand">
          <b>COMBATOS</b>
          <span>EDGE AUTONOMY STACK</span>
        </div>
        <nav className="modtabs">
          {MODULES.map((m, i) => (
            <div className={'mtab' + (i === 0 ? ' is-active' : '')} key={m}>
              <span className="mt-k">{String(i + 1).padStart(2, '0')}</span>
              <span className="mt-t">{m}</span>
            </div>
          ))}
        </nav>
        <div className="hero-status">
          <div className={'pill pill--deny'}>
            <i /><span>GPS</span><b>DENIED</b>
          </div>
          <div className={'pill pill--deny'}>
            <i /><span>LINK</span><b>NONE</b>
          </div>
          <div className={`pill ${t.tracking === 'OK' ? 'pill--ok' : 'pill--alert'}`}>
            <i /><span>STATE</span><b>{t.tracking === 'OK' ? 'LOCALIZED' : t.tracking}</b>
          </div>
          {t.wsConnected && (
            <div className="pill pill--ok">
              <i /><b>LIVE</b>
            </div>
          )}
          <div className="clk">
            <span className="ck">MISSION</span>
            <span className="cv">{clock}</span>
          </div>
        </div>
      </div>

      {/* body */}
      <div className="cmd-body">
        {/* telemetry rail */}
        <div className="pnl rail">
          <h4>FIELD TELEMETRY <em>LIVE</em></h4>

          <div className="sub-h">POSE · WORLD FRAME · M</div>
          <div className="readout">
            <div className="ro-row">
              <span className="lab">X · EAST</span>
              <span className={'num num--amber'}>{t.pose.x >= 0 ? '+' : ''}{t.pose.x.toFixed(2)}</span>
            </div>
            <div className="ro-row">
              <span className="lab">Y · NORTH</span>
              <span className="num">{t.pose.y >= 0 ? '+' : ''}{t.pose.y.toFixed(2)}</span>
            </div>
            <div className="ro-row">
              <span className="lab">Z · ALT</span>
              <span className="num">{t.pose.z >= 0 ? '+' : ''}{t.pose.z.toFixed(2)}</span>
            </div>
            <div className="ro-row">
              <span className="lab">YAW · °</span>
              <span className="num">{t.yaw.toFixed(1)}</span>
            </div>
          </div>

          <div className="sub-h">KINEMATICS</div>
          <div className="edge-row">
            <div className="top">
              <span className="lab">VELOCITY</span>
              <span className="num num--amber">{t.vel.toFixed(2)} m/s</span>
            </div>
            <Gauge value={t.vel} max={5} accent="amber" />
          </div>
          <div className="edge-row">
            <div className="top">
              <span className="lab">EST DRIFT</span>
              <span className="num">{t.drift.toFixed(2)} m</span>
            </div>
            <Gauge value={t.drift} max={2} accent={t.drift > 1.4 ? 'red' : 'green'} />
          </div>

          <div className="sub-h">EDGE COMPUTE</div>
          <div className="readout">
            <div className="ro-row">
              <span className="lab">VSLAM</span>
              <span className="num num--green">{t.slam} ms</span>
            </div>
            <div className="ro-row">
              <span className="lab">YOLO v8</span>
              <span className="num num--green">{t.yolo} ms</span>
            </div>
          </div>
          <div className="edge-row" style={{ marginTop: '11px' }}>
            <div className="top">
              <span className="lab">GPU LOAD</span>
              <span className="num num--amber">{t.gpu}%</span>
            </div>
            <Gauge value={t.gpu} max={100} accent={t.gpu > 90 ? 'red' : 'amber'} />
          </div>
          <div className="edge-row">
            <div className="top">
              <span className="lab">CORE TEMP</span>
              <span className="num">{t.temp}°C</span>
            </div>
            <Gauge value={t.temp} max={95} accent="green" />
          </div>

          <div className="rail-foot">
            <span className="rf-dot" /> ALL SUBSYSTEMS NOMINAL
          </div>
        </div>

        {/* top-left: SLAM 3D environment */}
        <div className={`pnl nav-col${fullscreenPanel === 'slam-map' ? ' panel-fullscreen' : ''}`}>
          <h4>
            <span>NAVIGATION · STEREO VSLAM</span>
            <span className="panel-head-actions">
              <em>6-DoF</em>
              <button
                type="button"
                className="panel-fullscreen-btn"
                onClick={() => setFullscreenPanel(fullscreenPanel === 'slam-map' ? null : 'slam-map')}
                aria-label={fullscreenPanel === 'slam-map' ? 'Restore SLAM map panel' : 'Fullscreen SLAM map panel'}
                title={fullscreenPanel === 'slam-map' ? 'Restore' : 'Fullscreen'}
              >
                {fullscreenPanel === 'slam-map' ? 'RESTORE' : 'FULL'}
              </button>
            </span>
          </h4>
          <div className="fig map" style={{ flex: 1 }}>
            <VslamScene points={t.slamPointCloud} path={t.slamPath} pose={vslamPose} />
            <div className="corner tl" /><div className="corner tr" />
            <div className="corner bl" /><div className="corner br" />
            <div className="fig-val">● LIVE · {t.slamPointCloud.length} MAP PTS</div>
            <div className="fig-cap">VSLAM MAP · /slam/odometry · /slam/path · /slam/point_cloud</div>
            <div className="fig-legend">
              <span><i className="lg" />SLAM PATH</span>
              <span><i className="lg dot" />EGO POSE</span>
              <span><i className="lg cloud" />MAP POINTS</span>
            </div>
          </div>
        </div>

        {/* bottom-left: SLAM keyframe */}
        <div className={`pnl slam-frame-col${fullscreenPanel === 'slam-keyframe' ? ' panel-fullscreen' : ''}`}>
          <h4>
            <span>SLAM KEYFRAME</span>
            <span className="panel-head-actions">
              <em>{annotatedFeed ? `#${annotatedFeed.seq}` : 'WAITING'}</em>
              <button
                type="button"
                className="panel-fullscreen-btn"
                onClick={() => setFullscreenPanel(fullscreenPanel === 'slam-keyframe' ? null : 'slam-keyframe')}
                aria-label={fullscreenPanel === 'slam-keyframe' ? 'Restore SLAM keyframe panel' : 'Fullscreen SLAM keyframe panel'}
                title={fullscreenPanel === 'slam-keyframe' ? 'Restore' : 'Fullscreen'}
              >
                {fullscreenPanel === 'slam-keyframe' ? 'RESTORE' : 'FULL'}
              </button>
            </span>
          </h4>
          <div className="slam-keyframe">
            {annotatedFeed ? (
              <LiveFrameCanvas frame={annotatedFeed} className="sw-canvas" fit="contain" />
            ) : (
              <div className="hatch" data-cap={'WAITING\n/slam/tracked_image/compressed'} />
            )}
          </div>
        </div>

        {/* top-right: YOLOX stream */}
        <div className="pnl feed-col">
          <h4>YOLOX VIDEO STREAM <em>{liveFeed ? `#${liveFeed.seq}` : 'WAITING'}</em></h4>
          <div className="feed" style={{ aspectRatio: liveFeedAspect }} onClick={onEnterOptic}>
            {liveFeed ? (
              <LiveFrameCanvas frame={liveFeed} className="slam-live-canvas" fit="contain" overlay={cameraOverlay} />
            ) : (
              <div className="subj-ph hatch" data-cap={'AWAITING\nCAMERA FEED'} />
            )}
            <div className="feed-tag"><i />{liveFeed ? `LIVE · ${liveFeed.width}×${liveFeed.height}` : 'NO SIGNAL'}</div>
            <div className="enter">
              <div className="ico">⤢</div>
              <div className="lbl">ENTER OPTIC FEED</div>
            </div>
          </div>
        </div>

        {/* bottom-right: detections list */}
        <div className="pnl intel-col">
          <h4>DETECTIONS <em>{tracked} ACTIVE</em></h4>
          <div className="dtable">
            <div className="dt-head">
              <span>TRACK ID</span>
              <span>CLASS</span>
              <span className="r">CONF</span>
              <span className="r">RNG·M</span>
              <span className="r">BRG</span>
              <span>STATUS</span>
            </div>
            <div className="dt-body">
              {t.dets.map(d => (
                <div
                  key={d.id}
                  className={'dt-row' + (d.tone ? ` dt-row--${d.tone}` : '')}
                  onClick={() => d.st === 'TRACK' && !d.confirmed && onConfirm(d.numericId, d.id)}
                  title={d.st === 'TRACK' && !d.confirmed ? 'Click to confirm target' : undefined}
                  style={{ cursor: d.st === 'TRACK' && !d.confirmed ? 'pointer' : 'default' }}
                >
                  <span className="dt-id">{d.id}</span>
                  <span>{d.cls}</span>
                  <span className="r mono">{d.conf.toFixed(2)}</span>
                  <span className="r mono">{isNaN(d.rng) ? '---' : d.rng.toFixed(1)}</span>
                  <span className="r mono">{isNaN(d.brg) ? '---' : String(d.brg).padStart(3, '0')}</span>
                  <span className="dt-st">{d.confirmed ? 'CONFIRMED' : d.st}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* log */}
      <div className="logbar">
        {log.slice(-4).map((e, i) => (
          <span className="le" key={i}>
            <span className="ts">{e.ts}</span>
            <b className={e.tone}>{e.src}</b>
            <span>{e.msg}</span>
          </span>
        ))}
      </div>
    </div>
  )
}
