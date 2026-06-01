/**
 * SwarmHuntView.tsx — the SWARM tab body for the Outcast Virus command dashboard.
 *
 * Left column: TRAIN (start/stop a MAPPO run on the hunt-and-seek 3D env, with
 * live reward / captures / coverage metrics) and RUN (launch the trained policy
 * in the live PyBullet environment). Right column: the live PyBullet sim feed
 * with observer / chase / FPV camera views (PyBulletSimPanel owns the cameras).
 *
 * Replaces the old blank SWARM · SECTOR A/B placeholder tiles.
 */

import { useCallback, useEffect, useState } from 'react'
import PyBulletSimPanel from '../panels/PyBulletSimPanel'
import { getScenarioDefaults } from '../gym/battlefieldParams'
import { useTraining } from '../gym/TrainingDashboard'
import { TrainingMetricsChart } from '../gym/TrainingMetricsChart'
import {
  DEFAULT_TRAIN_TIMESTEPS,
  decommissionPolicy,
  fetchPolicyDeployed,
  startPyBulletSim,
  stopPyBulletSim,
} from '../gym/trainApi'

const ENV_ID = 'hunt-and-seek'

export function SwarmHuntView() {
  const params = getScenarioDefaults(ENV_ID)
  const [deployed, setDeployed] = useState<boolean | null>(null)
  const refreshDeployed = useCallback(() => {
    void fetchPolicyDeployed(ENV_ID).then(setDeployed)
  }, [])
  const { status, metrics, history, start, stop } = useTraining(ENV_ID, params, {
    onComplete: refreshDeployed, // training auto-deploys (ONNX export) on finish
  })
  const [launching, setLaunching] = useState(false)
  const [launchError, setLaunchError] = useState<string | null>(null)
  const [simKey, setSimKey] = useState(0)
  const [running, setRunning] = useState(false)
  // Kill switch is two-stage: first press HALTS the live run (policy stays
  // deployed, so you can re-run without retraining). Second press while armed
  // DECOMMISSIONS the policy — then the only way back is a fresh retrain.
  const [killArmed, setKillArmed] = useState(false)

  const training = status === 'running'
  const simFrames = metrics && Number.isFinite(metrics.step) ? metrics.step : 0
  const trainProgress = DEFAULT_TRAIN_TIMESTEPS > 0
    ? Math.min(1, Math.max(0, simFrames / DEFAULT_TRAIN_TIMESTEPS))
    : 0
  const missionScore = metrics ? Math.round(Math.max(metrics.reward, metrics.task_score, 0) * 100) : null
  const interceptRate = metrics ? Math.round(Math.max(0, metrics.task_score) * 100) : null
  const coveragePct = metrics ? Math.round(Math.max(0, Math.min(1, metrics.coverage)) * 100) : null
  const policyIteration = metrics ? Math.max(1, metrics.episode).toLocaleString() : '—'
  const deployGate =
    status === 'completed' || deployed
      ? 'ARMED'
      : trainProgress >= 0.75
        ? 'FINAL'
        : trainProgress >= 0.35
          ? 'LEARNING'
          : 'WARMUP'

  useEffect(() => { refreshDeployed() }, [refreshDeployed])

  const launchSim = useCallback(async () => {
    setLaunching(true)
    setLaunchError(null)
    try {
      // Always runs the trained MAPPO policy in the environment.
      const res = await startPyBulletSim(ENV_ID, 'observer', 0, 'trained')
      if (!res.ok) setLaunchError(res.error ?? 'Failed to start sim')
      else {
        setRunning(true)
        setKillArmed(false) // fresh run — back to the first-press (halt) stage
        setSimKey((k) => k + 1) // remount the panel to reconnect cleanly
      }
    } catch (err) {
      setLaunchError(err instanceof Error ? err.message : 'Failed to start sim')
    } finally {
      setLaunching(false)
    }
  }, [])

  const killSwitch = useCallback(async () => {
    if (!killArmed) {
      // First press: just halt the live run. Policy stays deployed.
      await stopPyBulletSim()
      setRunning(false)
      setSimKey((k) => k + 1)
      setKillArmed(true)
      return
    }
    // Second press: decommission — requires a full retrain to run again.
    await decommissionPolicy(ENV_ID)
    setRunning(false)
    setDeployed(false)
    setSimKey((k) => k + 1)
    setKillArmed(false)
  }, [killArmed])

  return (
    <div className="cmd-body cmd-body--swarm cmd-body--hunt">
      {/* LEFT — controls */}
      <div className="pnl pnl-tile hunt-controls">
        <h4>
          <span>SWARM · HUNT &amp; SEEK · 3D</span>
          <span className={`hunt-status hunt-status--${status}`}>{status.toUpperCase()}</span>
        </h4>

        <div className="hunt-controls__scroll">
        <div className="hunt-section">
          <div className="hunt-section__head">
            <span>TRAIN</span>
            <em>MAPPO · CTDE · 3D</em>
          </div>
        
          <div className="hunt-btn-row">
            <button
              type="button"
              className="hunt-btn hunt-btn--primary"
              disabled={training}
              onClick={() => void start()}
            >
              {training ? 'Training…' : 'Start training'}
            </button>
            <button
              type="button"
              className="hunt-btn"
              disabled={!training}
              onClick={() => void stop()}
            >
              Stop
            </button>
          </div>

          <div className="hunt-metrics">
            <div className="hunt-metric">
              <span className="hunt-metric__lab">SIM FRAMES</span>
              <span className="hunt-metric__val">
                {metrics && Number.isFinite(metrics.step) ? metrics.step.toLocaleString() : '—'}
              </span>
            </div>
            <div className="hunt-metric">
              <span className="hunt-metric__lab">MISSION SCORE</span>
              <span className="hunt-metric__val">{missionScore == null ? '—' : `${missionScore}%`}</span>
            </div>
            <div className="hunt-metric">
              <span className="hunt-metric__lab">INTERCEPT RATE</span>
              <span className="hunt-metric__val">{interceptRate == null ? '—' : `${interceptRate}%`}</span>
            </div>
            <div className="hunt-metric">
              <span className="hunt-metric__lab">AO MAPPED</span>
              <span className="hunt-metric__val">{coveragePct == null ? '—' : `${coveragePct}%`}</span>
            </div>
            <div className="hunt-metric">
              <span className="hunt-metric__lab">POLICY ITERATION</span>
              <span className="hunt-metric__val">{policyIteration}</span>
            </div>
            <div className="hunt-metric">
              <span className="hunt-metric__lab">DEPLOY GATE</span>
              <span className="hunt-metric__val">{deployGate}</span>
            </div>
          </div>

          <div className="hunt-chart">
            <TrainingMetricsChart history={history} />
          </div>
        </div>

        <div className="hunt-section">
          <div className="hunt-section__head">
            <span>RUN IN ENVIRONMENT</span>
            <em>{deployed === false ? 'NOT DEPLOYED' : 'PyBullet · live'}</em>
          </div>
          <div className="hunt-btn-row">
            <button
              type="button"
              className="hunt-btn hunt-btn--primary"
              disabled={launching || deployed === false}
              onClick={() => void launchSim()}
            >
              {launching ? 'Starting…' : 'Run policy in environment'}
            </button>
          </div>
          {deployed === false ? (
            <p className="hunt-decom">
              POLICY DECOMMISSIONED — retrain &amp; deploy to run again.
            </p>
          ) : running ? (
            <p className="hunt-running">
              LIVE · <strong>Trained MAPPO</strong> policy in environment
            </p>
          ) : null}
          {launchError ? <p className="hunt-error">{launchError}</p> : null}

          <div className="hunt-btn-row">
            <button
              type="button"
              className="hunt-btn hunt-btn--danger"
              disabled={deployed === false}
              onClick={() => void killSwitch()}
            >
              {killArmed
                ? '⏻ Press again — decommission (retrain required)'
                : '⏻ Kill switch — halt run'}
            </button>
          </div>
          {killArmed ? (
            <p className="hunt-decom">
              RUN HALTED — policy still deployed. Re-run anytime, or press kill
              again to decommission (forces a retrain).
            </p>
          ) : null}
        </div>
        </div>
      </div>

      {/* RIGHT — live sim feed */}
      <div className="pnl pnl-tile hunt-feed">
        <h4>
          <span>SWARM · LIVE SIM FEED</span>
        </h4>
        <div className="hunt-feed__viewport">
          <PyBulletSimPanel
            key={simKey}
            envId={ENV_ID}
            missionName="Hunt &amp; Seek 3D"
          />
        </div>
      </div>
    </div>
  )
}
