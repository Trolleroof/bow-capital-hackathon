import { useMemo } from 'react'
import { LiveFrameCanvas } from './LiveFrameCanvas'
import { makeDetectionOverlay } from './OpticView'
import { VslamScene } from './VslamScene'
import type { CommandPanelId } from './panels'
import { PANEL_LABELS } from './panels'
import type { TelemetryState } from './useCombatState'
import { Gauge } from './atoms'

interface Props {
  panel: CommandPanelId
  t: TelemetryState
  onExit: () => void
  onConfirm: (numericId: number, id: string) => void
  onEnterOptic?: () => void
}

function fmtNum(n: number | null | undefined, digits = 2, signed = false) {
  if (n == null || !Number.isFinite(n)) return '---'
  const s = n.toFixed(digits)
  return signed && n >= 0 ? `+${s}` : s
}

function yawFromQuat(qw = 1, qz = 0, qx = 0, qy = 0) {
  return (Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) * (180 / Math.PI) + 360) % 360
}

function PanelHeader({ panel, meta, onExit }: { panel: CommandPanelId; meta?: string; onExit: () => void }) {
  return (
    <div className="expanded-panel__head">
      <button type="button" className="expanded-panel__exit" onClick={onExit}>
        EXIT {PANEL_LABELS[panel].split(' · ')[0]}
      </button>
      <div className="expanded-panel__title">
        <span>{PANEL_LABELS[panel]}</span>
        {meta ? <em>{meta}</em> : null}
      </div>
    </div>
  )
}

function TelemetryBody({ t }: { t: TelemetryState }) {
  const odom = t.slamOdometry
  const poseSource = odom ? 'SLAM ODOM' : t.hasNavFix ? 'NAV BUS' : null
  const poseX = poseSource ? (odom?.x ?? t.pose.x) : null
  const poseY = poseSource ? (odom?.y ?? t.pose.y) : null
  const poseZ = poseSource ? (odom?.z ?? t.pose.z) : null
  const poseYaw = poseSource
    ? (odom?.qw != null ? yawFromQuat(odom.qw, odom.qz ?? 0, odom.qx ?? 0, odom.qy ?? 0) : t.yaw)
    : null
  const mapPoints = t.slamPointCloudTotal || t.slamPointCloud.length
  const tracked = t.dets.filter(d => d.st !== 'LOST').length
  const confirmed = t.dets.filter(d => d.confirmed).length
  const primary = t.dets.find(d => d.tone === 'amber' || d.confirmed)
  const slamTone = t.slamStatus === 'OK' || t.slamStatus === 'TRACKING'
    ? 'green'
    : t.slamStatus === 'LOST'
      ? 'red'
      : ''

  return (
    <div className="expanded-panel__telemetry">
      <section>
        <div className="sub-h">POSE · {poseSource ?? 'NO FIX'} · M</div>
        <div className="readout">
          <div className="ro-row"><span className="lab">X · EAST</span><span className={'num' + (odom ? ' num--amber' : '')}>{fmtNum(poseX, 2, true)}</span></div>
          <div className="ro-row"><span className="lab">Y · NORTH</span><span className="num">{fmtNum(poseY, 2, true)}</span></div>
          <div className="ro-row"><span className="lab">Z · ALT</span><span className="num">{fmtNum(poseZ, 2, true)}</span></div>
          <div className="ro-row"><span className="lab">YAW · °</span><span className="num">{fmtNum(poseYaw, 1)}</span></div>
        </div>
      </section>
      {odom && (
        <section>
          <div className="sub-h">KINEMATICS · SLAM ODOM</div>
          <div className="edge-row">
            <div className="top">
              <span className="lab">SPEED</span>
              <span className="num num--amber">{fmtNum(t.vel, 2)} m/s</span>
            </div>
            <Gauge value={t.vel} max={5} accent="amber" />
          </div>
        </section>
      )}
      <section>
        <div className="sub-h">SLAM</div>
        <div className="readout">
          <div className="ro-row"><span className="lab">TRACK</span><span className={'num' + (slamTone ? ` num--${slamTone}` : '')}>{t.slamStatus !== '--' ? t.slamStatus : '---'}</span></div>
          <div className="ro-row"><span className="lab">PATH</span><span className="num">{t.slamPath.length > 0 ? `${t.slamPath.length} poses` : '---'}</span></div>
          <div className="ro-row"><span className="lab">MAP</span><span className="num num--amber">{mapPoints > 0 ? `${mapPoints} pts` : '---'}</span></div>
        </div>
      </section>
      <section>
        <div className="sub-h">TARGETING</div>
        <div className="readout">
          <div className="ro-row"><span className="lab">ACTIVE</span><span className={'num' + (tracked > 0 ? ' num--amber' : '')}>{tracked > 0 ? tracked : '---'}</span></div>
          <div className="ro-row"><span className="lab">LOCKED</span><span className="num">{primary?.id ?? '---'}</span></div>
          <div className="ro-row"><span className="lab">CONFIRMED</span><span className={'num' + (confirmed > 0 ? ' num--green' : '')}>{confirmed > 0 ? confirmed : '---'}</span></div>
        </div>
      </section>
      <section>
        <div className="sub-h">FEEDS</div>
        <div className="readout">
          <div className="ro-row"><span className="lab">CAMERA</span><span className={'num' + (t.cameraFrame ? ' num--green' : '')}>{t.cameraFrame ? `#${t.cameraFrame.seq}` : 'WAITING'}</span></div>
          <div className="ro-row"><span className="lab">SLAM FRAME</span><span className={'num' + (t.slamFrame ? ' num--green' : '')}>{t.slamFrame ? `#${t.slamFrame.seq}` : 'WAITING'}</span></div>
        </div>
      </section>
    </div>
  )
}

function DetectionsBody({ t, onConfirm }: { t: TelemetryState; onConfirm: (numericId: number, id: string) => void }) {
  const tracked = t.dets.filter(d => d.st !== 'LOST').length

  return (
    <div className="expanded-panel__detections dtable">
      <div className="dt-head">
        <span>TRACK ID</span>
        <span>CLASS</span>
        <span className="r">CONF</span>
        <span className="r">RNG·M</span>
        <span className="r">BRG</span>
        <span>IFF</span>
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
            <span className={d.allegiance ? `dt-iff dt-iff--${d.allegiance}` : 'dt-iff'}>
              {d.allegiance ? d.allegiance.toUpperCase() : '---'}
            </span>
            <span className="dt-st">{d.confirmed ? 'CONFIRMED' : d.st}</span>
          </div>
        ))}
        {t.dets.length === 0 && (
          <div className="expanded-panel__empty">NO DETECTIONS REPORTED</div>
        )}
      </div>
      <div className="expanded-panel__foot">{tracked} ACTIVE TARGET{tracked === 1 ? '' : 'S'}</div>
    </div>
  )
}

export function ExpandedPanelView({ panel, t, onExit, onConfirm, onEnterOptic }: Props) {
  const liveFeed = t.cameraFrame
  const annotatedFeed = t.slamFrame
  const cameraOverlay = useMemo(() => makeDetectionOverlay(t.dets, { lineScale: 0.42 }), [t.dets])
  const vslamPose = t.slamOdometry ?? { ...t.pose, qz: Math.sin(t.yaw * Math.PI / 360), qw: Math.cos(t.yaw * Math.PI / 360) }
  const tracked = t.dets.filter(d => d.st !== 'LOST').length

  const meta = panel === 'telemetry'
    ? (t.wsConnected ? 'LIVE' : 'OFFLINE')
    : panel === 'vslam'
      ? `${t.slamPointCloud.length} pts`
      : panel === 'keyframe'
        ? (annotatedFeed ? `#${annotatedFeed.seq}` : 'WAITING')
        : panel === 'targeting'
          ? (liveFeed ? `#${liveFeed.seq}` : 'WAITING')
          : panel === 'detections'
            ? `${tracked} ACTIVE`
            : undefined

  return (
    <div className={'expanded-panel expanded-panel--' + panel}>
      <PanelHeader panel={panel} meta={meta} onExit={onExit} />
      <div className="expanded-panel__body">
        {panel === 'telemetry' && <TelemetryBody t={t} />}
        {panel === 'vslam' && (
          <div className="fig map expanded-panel__scene">
            <VslamScene points={t.slamPointCloud} path={t.slamPath} pose={vslamPose} />
          </div>
        )}
        {panel === 'keyframe' && (
          <div className="slam-keyframe expanded-panel__frame">
            {annotatedFeed ? (
              <LiveFrameCanvas frame={annotatedFeed} className="sw-canvas" fit="contain" />
            ) : (
              <div className="hatch" data-cap={'WAITING\n/slam/tracked_image/compressed'} />
            )}
          </div>
        )}
        {panel === 'targeting' && (
          <div className="feed expanded-panel__feed">
            {liveFeed ? (
              <LiveFrameCanvas frame={liveFeed} className="slam-live-canvas" fit="contain" overlay={cameraOverlay} />
            ) : (
              <div className="subj-ph hatch" data-cap={'AWAITING\nCAMERA FEED'} />
            )}
            <div className="feed-tag"><i />{liveFeed ? `LIVE · ${liveFeed.width}×${liveFeed.height}` : 'NO SIGNAL'}</div>
            {onEnterOptic && liveFeed && (
              <button type="button" className="expanded-panel__optic-btn" onClick={onEnterOptic}>
                ENTER OPTIC HUD
              </button>
            )}
          </div>
        )}
        {panel === 'detections' && <DetectionsBody t={t} onConfirm={onConfirm} />}
        {(panel === 'swarm-a' || panel === 'swarm-b') && (
          <div className="hatch expanded-panel__blank" data-cap={'SWARM MODULE\nSTANDBY'} />
        )}
      </div>
    </div>
  )
}
