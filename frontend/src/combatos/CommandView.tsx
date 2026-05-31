import { useMemo, useState } from 'react'
import { LiveFrameCanvas } from './LiveFrameCanvas'
import type { TelemetryState, LogEntry } from './useCombatState'
import { Gauge } from './atoms'
import { VslamScene } from './VslamScene'
import { makeDetectionOverlay } from './OpticView'
import type { CommandPanelId } from './panels'

interface Props {
  t: TelemetryState
  log: LogEntry[]
  onEnterOptic: () => void
  onConfirm: (numericId: number, id: string) => void
  onExpandPanel: (panel: CommandPanelId) => void
}

function PanelExpandBtn({ panel, onExpandPanel }: { panel: CommandPanelId; onExpandPanel: (panel: CommandPanelId) => void }) {
  return (
    <button
      type="button"
      className="panel-fullscreen-btn"
      onClick={() => onExpandPanel(panel)}
      aria-label={`Expand ${panel} panel`}
      title="Fullscreen"
    >
      FULL
    </button>
  )
}

type CommandTab = 'swarm' | 'drone'

function fmtNum(n: number | null | undefined, digits = 2, signed = false) {
  if (n == null || !Number.isFinite(n)) return '---'
  const s = n.toFixed(digits)
  return signed && n >= 0 ? `+${s}` : s
}

function yawFromQuat(qw = 1, qz = 0, qx = 0, qy = 0) {
  return (Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) * (180 / Math.PI) + 360) % 360
}

export function CommandView({ t, log, onEnterOptic, onConfirm, onExpandPanel }: Props) {
  const [activeTab, setActiveTab] = useState<CommandTab>('drone')
  const clock = 'T+' + String(Math.floor(t.sec / 60)).padStart(2, '0') + ':' + String(t.sec % 60).padStart(2, '0')
  const tracked = t.dets.filter(d => d.st !== 'LOST').length
  const confirmed = t.dets.filter(d => d.confirmed).length
  const primary = t.dets.find(d => d.tone === 'amber' || d.confirmed)
  const liveFeed = t.cameraFrame
  const annotatedFeed = t.slamFrame
  const cameraOverlay = useMemo(() => makeDetectionOverlay(t.dets, { lineScale: 0.42 }), [t.dets])
  const vslamPose = t.slamOdometry ?? { ...t.pose, qz: Math.sin(t.yaw * Math.PI / 360), qw: Math.cos(t.yaw * Math.PI / 360) }

  const odom = t.slamOdometry
  const poseSource = odom ? 'SLAM ODOM' : t.hasNavFix ? 'NAV BUS' : null
  const poseX = poseSource ? (odom?.x ?? t.pose.x) : null
  const poseY = poseSource ? (odom?.y ?? t.pose.y) : null
  const poseZ = poseSource ? (odom?.z ?? t.pose.z) : null
  const poseYaw = poseSource
    ? (odom?.qw != null ? yawFromQuat(odom.qw, odom.qz ?? 0, odom.qx ?? 0, odom.qy ?? 0) : t.yaw)
    : null
  const mapPoints = t.slamPointCloudTotal || t.slamPointCloud.length
  const slamTone = t.slamStatus === 'OK' || t.slamStatus === 'TRACKING'
    ? 'green'
    : t.slamStatus === 'LOST'
      ? 'red'
      : ''
  const railFoot = !t.wsConnected
    ? { tone: 'red' as const, label: 'BUS OFFLINE' }
    : t.slamStatus === 'LOST'
      ? { tone: 'red' as const, label: 'SLAM LOST' }
      : tracked > 0
        ? { tone: 'amber' as const, label: `${tracked} TARGET${tracked === 1 ? '' : 'S'} LIVE` }
        : t.slamStatus !== '--'
          ? { tone: slamTone === 'green' ? 'green' as const : '' as const, label: `SLAM · ${t.slamStatus}` }
          : { tone: '' as const, label: 'AWAITING TELEMETRY' }

  return (
    <div className="cmd">
      {/* hero bar */}
      <div className="hero">
        <div className="brand">
          <b>COMBATOS</b>
          <span>EDGE AUTONOMY STACK</span>
        </div>
        <nav className="modtabs" role="tablist" aria-label="Command mode">
          <div
            role="tab"
            aria-selected={activeTab === 'swarm'}
            tabIndex={activeTab === 'swarm' ? 0 : -1}
            className={'mtab' + (activeTab === 'swarm' ? ' is-active' : '')}
            onClick={() => setActiveTab('swarm')}
          >
            <span className="mt-t">Swarm</span>
          </div>
          <div
            role="tab"
            aria-selected={activeTab === 'drone'}
            tabIndex={activeTab === 'drone' ? 0 : -1}
            className={'mtab' + (activeTab === 'drone' ? ' is-active' : '')}
            onClick={() => setActiveTab('drone')}
          >
            <span className="mt-t">Drone</span>
          </div>
        </nav>
        <div className="hero-status">
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
      {activeTab === 'swarm' ? (
        <div className="cmd-body cmd-body--swarm">
          <div className="pnl pnl-tile swarm-col swarm-col--a">
            <h4>
              <span>SWARM · SECTOR A</span>
              <span className="panel-head-actions">
                <PanelExpandBtn panel="swarm-a" onExpandPanel={onExpandPanel} />
              </span>
            </h4>
            <div className="hatch swarm-blank" />
          </div>
          <div className="pnl pnl-tile swarm-col swarm-col--b">
            <h4>
              <span>SWARM · SECTOR B</span>
              <span className="panel-head-actions">
                <PanelExpandBtn panel="swarm-b" onExpandPanel={onExpandPanel} />
              </span>
            </h4>
            <div className="hatch swarm-blank" />
          </div>
        </div>
      ) : (
      <div className="cmd-body">
        {/* telemetry rail */}
        <div className="pnl rail">
          <h4>
            <span>FIELD TELEMETRY <em>{t.wsConnected ? 'LIVE' : 'OFFLINE'}</em></span>
            <span className="panel-head-actions">
              <PanelExpandBtn panel="telemetry" onExpandPanel={onExpandPanel} />
            </span>
          </h4>

          <div className="sub-h">POSE · {poseSource ?? 'NO FIX'} · M</div>
          <div className="readout">
            <div className="ro-row">
              <span className="lab">X · EAST</span>
              <span className={'num' + (odom ? ' num--amber' : '')}>{fmtNum(poseX, 2, true)}</span>
            </div>
            <div className="ro-row">
              <span className="lab">Y · NORTH</span>
              <span className="num">{fmtNum(poseY, 2, true)}</span>
            </div>
            <div className="ro-row">
              <span className="lab">Z · ALT</span>
              <span className="num">{fmtNum(poseZ, 2, true)}</span>
            </div>
            <div className="ro-row">
              <span className="lab">YAW · °</span>
              <span className="num">{fmtNum(poseYaw, 1)}</span>
            </div>
          </div>

          {odom && (
            <>
              <div className="sub-h">KINEMATICS · SLAM ODOM</div>
              <div className="edge-row">
                <div className="top">
                  <span className="lab">SPEED</span>
                  <span className="num num--amber">{fmtNum(t.vel, 2)} m/s</span>
                </div>
                <Gauge value={t.vel} max={5} accent="amber" />
              </div>
            </>
          )}

          <div className="sub-h">SLAM</div>
          <div className="readout">
            <div className="ro-row">
              <span className="lab">TRACK</span>
              <span className={'num' + (slamTone ? ` num--${slamTone}` : '')}>
                {t.slamStatus !== '--' ? t.slamStatus : '---'}
              </span>
            </div>
            <div className="ro-row">
              <span className="lab">PATH</span>
              <span className="num">{t.slamPath.length > 0 ? `${t.slamPath.length} poses` : '---'}</span>
            </div>
            <div className="ro-row">
              <span className="lab">MAP</span>
              <span className="num num--amber">{mapPoints > 0 ? `${mapPoints} pts` : '---'}</span>
            </div>
            {t.wsConnected && (
              <div className="ro-row">
                <span className="lab">DROP</span>
                <span className={'num' + (t.slamDiagnostics.droppedFrames > 0 ? ' num--red' : '')}>
                  {t.slamDiagnostics.droppedFrames}
                </span>
              </div>
            )}
          </div>

          <div className="sub-h">TARGETING</div>
          <div className="readout">
            <div className="ro-row">
              <span className="lab">ACTIVE</span>
              <span className={'num' + (tracked > 0 ? ' num--amber' : '')}>{tracked > 0 ? tracked : '---'}</span>
            </div>
            <div className="ro-row">
              <span className="lab">LOCKED</span>
              <span className="num">{primary?.id ?? '---'}</span>
            </div>
            <div className="ro-row">
              <span className="lab">CONFIRMED</span>
              <span className={'num' + (confirmed > 0 ? ' num--green' : '')}>{confirmed > 0 ? confirmed : '---'}</span>
            </div>
          </div>

          <div className="sub-h">FEEDS</div>
          <div className="readout">
            <div className="ro-row">
              <span className="lab">CAMERA</span>
              <span className={'num' + (liveFeed ? ' num--green' : '')}>
                {liveFeed ? `#${liveFeed.seq}` : 'WAITING'}
              </span>
            </div>
            <div className="ro-row">
              <span className="lab">SLAM FRAME</span>
              <span className={'num' + (annotatedFeed ? ' num--green' : '')}>
                {annotatedFeed ? `#${annotatedFeed.seq}` : 'WAITING'}
              </span>
            </div>
          </div>

          <div className="rail-foot">
            <span className={'rf-dot' + (railFoot.tone ? ` rf-dot--${railFoot.tone}` : '')} /> {railFoot.label}
          </div>
        </div>

        {/* top-left: SLAM 3D environment */}
        <div className="pnl pnl-tile nav-col">
          <h4>
            <span>NAVIGATION · STEREO VSLAM</span>
            <span className="panel-head-actions">
              <em>{t.slamPointCloud.length} pts</em>
              <PanelExpandBtn panel="vslam" onExpandPanel={onExpandPanel} />
            </span>
          </h4>
          <div className="fig map">
            <VslamScene points={t.slamPointCloud} path={t.slamPath} pose={vslamPose} />
          </div>
        </div>

        {/* bottom-left: SLAM keyframe */}
        <div className="pnl pnl-tile slam-frame-col">
          <h4>
            <span>SLAM KEYFRAME</span>
            <span className="panel-head-actions">
              <em>{annotatedFeed ? `#${annotatedFeed.seq}` : 'WAITING'}</em>
              <PanelExpandBtn panel="keyframe" onExpandPanel={onExpandPanel} />
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
        <div className="pnl pnl-tile feed-col">
          <h4>
            <span>Targeting <em>{liveFeed ? `#${liveFeed.seq}` : 'WAITING'}</em></span>
            <span className="panel-head-actions">
              <PanelExpandBtn panel="targeting" onExpandPanel={onExpandPanel} />
            </span>
          </h4>
          <div className="feed" onClick={onEnterOptic}>
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
        <div className="pnl pnl-tile intel-col">
          <h4>
            <span>DETECTIONS <em>{tracked} ACTIVE</em></span>
            <span className="panel-head-actions">
              <PanelExpandBtn panel="detections" onExpandPanel={onExpandPanel} />
            </span>
          </h4>
          <div className="dtable">
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
            </div>
          </div>
        </div>
      </div>
      )}

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
