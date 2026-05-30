import { useEffect, useRef, useState } from 'react'
import './App.css'
import CompositeScenePanel from './panels/CompositeScenePanel'
import GymScenarioStage, { ScenarioMiniPreview } from './gym/GymScenarioStage'
import { getScenarioById, scenarios } from './gym/scenarios'
import { setActiveEnvId, type PolicyStatus } from './swarm/policy'

// ─── Routing ─────────────────────────────────────────────────────────────────

type AppRoute = 'menu' | 'gym' | 'sim'

function parseRoute(): AppRoute {
  const raw = window.location.hash.replace(/^#/, '')
  if (raw === 'sim') return 'sim'
  if (raw === 'gym') return 'gym'
  return 'menu'
}

// ─── App ─────────────────────────────────────────────────────────────────────

function App() {
  const [route, setRoute] = useState<AppRoute>(parseRoute)
  const [selectedEnvId, setSelectedEnvId] = useState(scenarios[0].id)
  const [simEnvId, setSimEnvId] = useState(scenarios[0].id)

  const [policyStore, setPolicyStore] = useState<Record<string, PolicyStatus>>(
    () =>
      Object.fromEntries(
        scenarios.map((s) => [s.id, 'not-trained' as PolicyStatus]),
      ),
  )

  const [toast, setToast] = useState<string | null>(null)
  const toastTimer = useRef<number | null>(null)

  // ── Hash routing ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!window.location.hash || window.location.hash === '#') {
      window.location.hash = 'menu'
    }
    const onHashChange = () => setRoute(parseRoute())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  // ── Derived ───────────────────────────────────────────────────────────────
  const hasAnyReadyPolicy = Object.values(policyStore).some((s) => s === 'ready')

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handlePolicyReady = (envId: string) => {
    setPolicyStore((prev) => ({ ...prev, [envId]: 'ready' }))
    const name = getScenarioById(envId).name
    showToast(`${name}: policy exported — Mission Sim unlocked`)
  }

  const handleTrainingStart = (envId: string) => {
    setPolicyStore((prev) => ({ ...prev, [envId]: 'training' }))
  }

  const handleTrainingError = (msg: string) => {
    showToast(msg)
  }

  const gymStageProps = {
    onPolicyReady: handlePolicyReady,
    onTrainingStart: handleTrainingStart,
    onTrainingError: handleTrainingError,
  }

  function showToast(msg: string) {
    setToast(msg)
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast(null), 4000)
  }

  const enterGym = (envId: string) => {
    setSelectedEnvId(envId)
    window.location.hash = 'gym'
    setRoute('gym')
  }

  const goToMenu = () => {
    window.location.hash = 'menu'
    setRoute('menu')
  }

  const goToSim = () => {
    const chosenEnvId =
      policyStore[selectedEnvId] === 'ready'
        ? selectedEnvId
        : (scenarios.find((s) => policyStore[s.id] === 'ready')?.id ?? selectedEnvId)

    setActiveEnvId(chosenEnvId)
    setSimEnvId(chosenEnvId)
    window.location.hash = 'sim'
    setRoute('sim')
  }

  // ── Shared top-right nav ──────────────────────────────────────────────────
  const nav = (
    <nav className="top-nav" aria-label="Primary">
      <button
        type="button"
        className={route !== 'sim' ? 'is-active' : undefined}
        onClick={goToMenu}
      >
        Gym
      </button>
      <button
        type="button"
        className={route === 'sim' ? 'is-active' : undefined}
        onClick={goToSim}
      >
        Mission Sim
      </button>
    </nav>
  )

  // ── Render: Mission Sim ───────────────────────────────────────────────────
  if (route === 'sim') {
    const simEnv = getScenarioById(simEnvId)
    const policyReady = policyStore[simEnvId] === 'ready'

    return (
      <main className="app-shell app-shell--sim">
        <div className="app-backdrop" />
        <section className="sim-viewport-full" aria-label="Mission simulation">
          <div className="sim-viewport-nav">{nav}</div>
          <div className="sim-policy-badge" aria-label="Active policy info">
            <span className="sim-policy-env">{simEnv.name}</span>
            <em data-status={policyReady ? 'Ready' : 'Queued'}>
              {policyReady ? 'ONNX' : hasAnyReadyPolicy ? 'ONNX' : 'Policy required'}
            </em>
          </div>
          <CompositeScenePanel
            key={simEnvId}
            missionName={simEnv.name}
            policyEnabled={policyReady}
          />
        </section>
        {toast && <div className="app-toast" role="status">{toast}</div>}
      </main>
    )
  }

  // ── Render: Gym training view ─────────────────────────────────────────────
  if (route === 'gym') {
    const env = getScenarioById(selectedEnvId)
    const status = policyStore[env.id]

    return (
      <main className="app-shell app-shell--gym-full">
        <div className="app-backdrop" />
        <section className="gym-scene-full" aria-label={`${env.name} training environment`}>
          <div className="gym-scene-full-nav">
            <button type="button" className="gym-back-btn" onClick={goToMenu}>
              ← Environments
            </button>
            <div className="gym-scene-full-meta">
              <span className="gym-scene-full-name">{env.name}</span>
              <em data-status={status === 'ready' ? 'Ready' : status === 'training' ? 'Queued' : env.status}>
                {status === 'ready' ? 'Policy ready' : status === 'training' ? 'Training…' : env.label}
              </em>
            </div>
            <div className="gym-scene-full-right">{nav}</div>
          </div>
          <GymScenarioStage key={env.id} scenario={env} {...gymStageProps} />
        </section>
        {toast && <div className="app-toast" role="status">{toast}</div>}
      </main>
    )
  }

  // ── Render: Menu screen (default) ─────────────────────────────────────────
  return (
    <main className="app-shell app-shell--menu">
      <div className="app-backdrop" />

      <div className="menu-viewport">
        <header className="menu-header">
          <div className="menu-header-left">
            <h1 className="menu-title">CombatOS</h1>
            <span className="menu-subtitle">SWARM TRAINING GYMNASIUM</span>
          </div>
          <div className="menu-header-right">{nav}</div>
        </header>

        <section className="menu-kicker" aria-label="Section label">
          Select Training Environment
        </section>

        <div className="menu-grid" role="list" aria-label="Training scenarios">
          {scenarios.map((s) => {
            const status = policyStore[s.id]
            const isReady = status === 'ready'
            const isTraining = status === 'training'

            return (
              <button
                key={s.id}
                type="button"
                role="listitem"
                className={`menu-card ${isReady ? 'menu-card--ready' : ''}`}
                onClick={() => enterGym(s.id)}
                aria-label={`Enter ${s.name}`}
              >
                <div className="menu-card-preview">
                  <ScenarioMiniPreview scenarioId={s.id} />
                </div>
                <div className="menu-card-topline">
                  <strong className="menu-card-name">{s.name}</strong>
                  <span
                    className="menu-card-badge"
                    data-status={
                      isReady ? 'Ready' : isTraining ? 'Queued' : s.status
                    }
                  >
                    {isReady ? 'Policy ready' : isTraining ? 'Training…' : s.label}
                  </span>
                </div>
                <p className="menu-card-summary">{s.summary}</p>
                <div className="menu-card-meta">
                  <span className="menu-card-meta-item">
                    <span className="menu-card-meta-label">Reward</span>
                    {s.telemetryLabels[0]}
                  </span>
                  <span className="menu-card-meta-item">
                    <span className="menu-card-meta-label">Track</span>
                    {s.telemetryLabels[1]}
                  </span>
                  <span className="menu-card-meta-item">
                    <span className="menu-card-meta-label">Objective</span>
                    {s.telemetryLabels[2]}
                  </span>
                </div>
                <div className="menu-card-cta">
                  {isTraining ? 'View Training →' : 'Train →'}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {toast && <div className="app-toast" role="status">{toast}</div>}
    </main>
  )
}

export default App
