import { useEffect, useRef, useState } from 'react'
import './App.css'
import CompositeScenePanel from './panels/CompositeScenePanel'
import GymScenarioStage, { ScenarioMiniPreview } from './gym/GymScenarioStage'
import { getScenarioById, scenarios } from './gym/scenarios'
import { checkPolicyExists, type PolicyStatus } from './swarm/policy'

type AppRoute =
  | { view: 'menu' }
  | { view: 'gym'; envId: string }
  | { view: 'sim'; envId: string }

function scenarioExists(envId: string): boolean {
  return scenarios.some((scenario) => scenario.id === envId)
}

function parseRoute(): AppRoute {
  const raw = window.location.hash.replace(/^#/, '')
  if (raw.startsWith('gym/')) {
    const envId = raw.slice(4)
    return { view: 'gym', envId: scenarioExists(envId) ? envId : scenarios[0].id }
  }
  if (raw.startsWith('sim/')) {
    const envId = raw.slice(4)
    return { view: 'sim', envId: scenarioExists(envId) ? envId : scenarios[0].id }
  }
  return { view: 'menu' }
}

function setHashRoute(hash: string) {
  window.history.pushState(null, '', `#${hash}`)
}

function App() {
  const [route, setRoute] = useState<AppRoute>(parseRoute)
  const [policyStore, setPolicyStore] = useState<Record<string, PolicyStatus>>(
    () =>
      Object.fromEntries(
        scenarios.map((scenario) => [scenario.id, 'not-trained' as PolicyStatus]),
      ),
  )
  const [toast, setToast] = useState<string | null>(null)
  const toastTimer = useRef<number | null>(null)

  useEffect(() => {
    if (!window.location.hash || window.location.hash === '#') {
      window.location.hash = 'menu'
    }
    const onHashChange = () => setRoute(parseRoute())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  useEffect(() => {
    let cancelled = false
    void Promise.all(
      scenarios.map(async (scenario) => [scenario.id, await checkPolicyExists(scenario.id)] as const),
    ).then(results => {
      if (cancelled) return
      setPolicyStore(prev => {
        const next = { ...prev }
        for (const [envId, exists] of results) {
          if (exists && next[envId] === 'not-trained') next[envId] = 'ready'
        }
        return next
      })
    })
    return () => {
      cancelled = true
    }
  }, [])

  function showToast(message: string) {
    setToast(message)
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast(null), 4000)
  }

  const enterMenu = () => {
    setHashRoute('menu')
    setRoute({ view: 'menu' })
  }

  const enterGym = (envId: string) => {
    setHashRoute(`gym/${envId}`)
    setRoute({ view: 'gym', envId })
  }

  const enterSim = (envId: string) => {
    setHashRoute(`sim/${envId}`)
    setRoute({ view: 'sim', envId })
  }

  const handlePolicyReady = (envId: string) => {
    setPolicyStore((prev) => ({ ...prev, [envId]: 'ready' }))
    showToast(`${getScenarioById(envId).name}: policy exported`)
  }

  const handleTrainingStart = (envId: string) => {
    setPolicyStore((prev) => ({ ...prev, [envId]: 'training' }))
  }

  const handleTrainingError = (message: string) => {
    showToast(message)
  }

  const scopedNav = (envId: string, active: 'gym' | 'sim') => (
    <nav className="top-nav" aria-label="Primary">
      <button
        type="button"
        className={active === 'gym' ? 'is-active' : undefined}
        onClick={() => enterGym(envId)}
      >
        Gym
      </button>
      <button
        type="button"
        className={active === 'sim' ? 'is-active' : undefined}
        onClick={() => enterSim(envId)}
      >
        Mission Sim
      </button>
    </nav>
  )

  if (route.view === 'sim') {
    const env = getScenarioById(route.envId)
    const policyReady = policyStore[route.envId] === 'ready'

    return (
      <main className="app-shell app-shell--sim">
        <div className="app-backdrop" />
        <section className="sim-viewport-full" aria-label={`${env.name} mission simulation`}>
          <header className="sim-top-bar">
            <div className="sim-top-bar__left">
              <div className="sim-policy-badge" aria-label="Active policy info">
                <span className="sim-policy-badge__kicker">Mission</span>
                <span className="sim-policy-env">{env.name}</span>
                <em data-status={policyReady ? 'Ready' : 'Queued'}>
                  {policyReady ? 'ONNX' : 'Policy required'}
                </em>
              </div>
              <div className="sim-comms-badges" aria-label="Comms status">
                <span>GPS: DENIED</span>
                <span>LINK: NONE</span>
                <span>LOCALIZED</span>
              </div>
            </div>
            <div className="sim-viewport-nav">{scopedNav(route.envId, 'sim')}</div>
          </header>
          <CompositeScenePanel
            key={route.envId}
            envId={route.envId}
            missionName={env.name}
            policyEnabled={policyReady}
          />
        </section>
        {toast && <div className="app-toast" role="status">{toast}</div>}
      </main>
    )
  }

  if (route.view === 'gym') {
    const env = getScenarioById(route.envId)
    const status = policyStore[route.envId]

    return (
      <main className="app-shell app-shell--gym-full">
        <div className="app-backdrop" />
        <section className="gym-scene-full" aria-label={`${env.name} training environment`}>
          <div className="gym-scene-full-nav">
            <button type="button" className="gym-back-btn" onClick={enterMenu}>
              ← Environments
            </button>
            <div className="gym-scene-full-meta">
              <span className="gym-scene-full-name">{env.name}</span>
              <em data-status={status === 'ready' ? 'Ready' : status === 'training' ? 'Queued' : env.status}>
                {status === 'ready' ? 'Policy ready' : status === 'training' ? 'Training...' : env.label}
              </em>
            </div>
            <div className="gym-scene-full-right">{scopedNav(route.envId, 'gym')}</div>
          </div>
          <GymScenarioStage
            key={route.envId}
            scenario={env}
            policyStatus={status}
            onPolicyReady={handlePolicyReady}
            onTrainingStart={handleTrainingStart}
            onTrainingError={handleTrainingError}
          />
        </section>
        {toast && <div className="app-toast" role="status">{toast}</div>}
      </main>
    )
  }

  return (
    <main className="app-shell app-shell--menu">
      <div className="app-backdrop" />
      <div className="menu-viewport">
        <header className="menu-header">
          <div className="menu-header-left">
            <h1 className="menu-title">CombatOS</h1>
            <span className="menu-subtitle">SWARM TRAINING GYMNASIUM</span>
          </div>
        </header>

        <section className="menu-kicker" aria-label="Section label">
          Select Training Environment
        </section>

        <div className="menu-grid" role="list" aria-label="Training scenarios">
          {scenarios.map((scenario) => {
            const status = policyStore[scenario.id]
            const isReady = status === 'ready'
            const isTraining = status === 'training'

            return (
              <button
                key={scenario.id}
                type="button"
                role="listitem"
                className={`menu-card ${isReady ? 'menu-card--ready' : ''}`}
                onClick={() => enterGym(scenario.id)}
                aria-label={`Enter ${scenario.name}`}
              >
                <div className="menu-card-preview">
                  <ScenarioMiniPreview scenarioId={scenario.id} />
                </div>
                <div className="menu-card-topline">
                  <strong className="menu-card-name">{scenario.name}</strong>
                  <span
                    className="menu-card-badge"
                    data-status={isReady ? 'Ready' : isTraining ? 'Queued' : scenario.status}
                  >
                    {isReady ? 'Policy ready' : isTraining ? 'Training...' : scenario.label}
                  </span>
                </div>
                <p className="menu-card-summary">{scenario.summary}</p>
                <div className="menu-card-meta">
                  <span className="menu-card-meta-item">
                    <span className="menu-card-meta-label">Reward</span>
                    {scenario.telemetryLabels[0]}
                  </span>
                  <span className="menu-card-meta-item">
                    <span className="menu-card-meta-label">Track</span>
                    {scenario.telemetryLabels[1]}
                  </span>
                  <span className="menu-card-meta-item">
                    <span className="menu-card-meta-label">Objective</span>
                    {scenario.telemetryLabels[2]}
                  </span>
                </div>
                <div className="menu-card-cta">
                  {isTraining ? 'View Training ->' : 'Train ->'}
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
