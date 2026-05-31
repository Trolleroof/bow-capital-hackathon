import { useEffect, useRef, useState } from 'react'
import './App.css'
import CompositeScenePanel from './panels/CompositeScenePanel'
import GymScenarioStage, { ScenarioMiniPreview } from './gym/GymScenarioStage'
import { getScenarioById, scenarios } from './gym/scenarios'
import { CombatOS } from './combatos/CombatOS'
import {
  checkPolicyExists,
  setActiveEnvId,
  type PolicyStatus,
} from './swarm/policy'

const ALLOW_HEURISTIC = true

type AppRoute =
  | { view: 'menu' }
  | { view: 'gym'; envId: string }
  | { view: 'sim'; envId: string }
  | { view: 'combatos' }

const scenarioExists = (envId: string) => getScenarioById(envId) != null

const parseRoute = (): AppRoute => {
  const hash = window.location.hash.replace(/^#\/?/, '')
  const [view, envId] = hash.split('/')

  if (view === 'combatos') return { view: 'combatos' }

  if (view === 'gym' && envId && scenarioExists(envId)) {
    return { view: 'gym', envId }
  }

  if (view === 'sim' && envId && scenarioExists(envId)) {
    return { view: 'sim', envId }
  }

  if (view === 'sim') {
    return { view: 'sim', envId: scenarios[0].id }
  }

  return { view: 'menu' }
}

const setHashRoute = (hash: string) => {
  window.history.pushState(null, '', `#${hash}`)
}

export default function App() {
  const [route, setRoute] = useState<AppRoute>(() => parseRoute())
  const [policyStore, setPolicyStore] = useState<Record<string, PolicyStatus>>(
    () =>
      Object.fromEntries(
        scenarios.map((scenario) => [
          scenario.id,
          'not-trained' as PolicyStatus,
        ]),
      ) as Record<string, PolicyStatus>,
  )
  const [toast, setToast] = useState<string | null>(null)
  const toastTimer = useRef<number | null>(null)

  useEffect(() => {
    if (!window.location.hash) {
      setHashRoute('menu')
    }

    const onHashChange = () => setRoute(parseRoute())
    window.addEventListener('hashchange', onHashChange)
    window.addEventListener('popstate', onHashChange)

    return () => {
      window.removeEventListener('hashchange', onHashChange)
      window.removeEventListener('popstate', onHashChange)
    }
  }, [])

  useEffect(() => {
    let canceled = false

    Promise.all(
      scenarios.map(async (scenario) => {
        const exists = await checkPolicyExists(scenario.id)
        const status: PolicyStatus = exists ? 'ready' : 'not-trained'
        return [scenario.id, status] as const
      }),
    ).then((entries) => {
      if (!canceled) {
        setPolicyStore(Object.fromEntries(entries) as Record<string, PolicyStatus>)
      }
    })

    return () => {
      canceled = true
    }
  }, [])

  const showToast = (message: string) => {
    setToast(message)

    if (toastTimer.current != null) {
      window.clearTimeout(toastTimer.current)
    }

    toastTimer.current = window.setTimeout(() => setToast(null), 2600)
  }

  const simAllowedFor = (envId: string) =>
    policyStore[envId] === 'ready' || ALLOW_HEURISTIC

  const enterMenu = () => {
    setHashRoute('menu')
    setRoute({ view: 'menu' })
  }

  const enterGym = (envId: string) => {
    setHashRoute(`gym/${envId}`)
    setRoute({ view: 'gym', envId })
  }

  const enterSim = (envId: string) => {
    if (!simAllowedFor(envId)) {
      showToast('Train this environment first to unlock Mission Sim.')
      return
    }

    setActiveEnvId(envId)
    setHashRoute(`sim/${envId}`)
    setRoute({ view: 'sim', envId })
  }

  const enterCombatOS = () => {
    setHashRoute('combatos')
    setRoute({ view: 'combatos' })
  }

  const handlePolicyReady = (envId: string) => {
    setPolicyStore((current) => ({ ...current, [envId]: 'ready' }))
  }

  const handleTrainingStart = (envId: string) => {
    setPolicyStore((current) => ({ ...current, [envId]: 'training' }))
  }

  const handleTrainingError = (message: string) => {
    showToast(message)
  }

  const scopedNav = (envId: string, active: 'gym' | 'sim') => {
    const simLocked = !simAllowedFor(envId)

    return (
      <div className="app-nav app-nav--scoped" aria-label="Scenario navigation">
        <button
          type="button"
          className={`nav-pill ${active === 'gym' ? 'nav-pill--active' : ''}`}
          onClick={() => enterGym(envId)}
        >
          Gym
        </button>
        <button
          type="button"
          className={`nav-pill ${active === 'sim' ? 'nav-pill--active' : ''}`}
          onClick={() => enterSim(envId)}
          disabled={simLocked}
          title={simLocked ? 'Train this environment first.' : undefined}
        >
          Mission Sim
        </button>
        <button type="button" className="nav-pill" onClick={enterCombatOS}>
          CombatOS
        </button>
      </div>
    )
  }

  if (route.view === 'combatos') {
    return <CombatOS />
  }

  if (route.view === 'sim') {
    const env = getScenarioById(route.envId) ?? scenarios[0]
    const policyReady = policyStore[env.id] === 'ready'
    const controllerActive = policyReady || ALLOW_HEURISTIC

    return (
      <main className="app-shell app-shell--sim">
        <div className="sim-top-bar">
          <button type="button" className="sim-back-link" onClick={enterMenu}>
            Back to gym menu
          </button>
          <div className="sim-title-block">
            <span>Mission Sim</span>
            <strong>{env.name}</strong>
          </div>
          {scopedNav(env.id, 'sim')}
          <div
            className={`sim-policy-badge ${
              controllerActive ? 'sim-policy-badge--ready' : ''
            }`}
          >
            <span className="sim-policy-badge__kicker">Controller</span>
            <span>
              {policyReady
                ? 'Trained policy'
                : ALLOW_HEURISTIC
                  ? 'Task behavior'
                  : 'Locked'}
            </span>
          </div>
        </div>

        {!policyReady && ALLOW_HEURISTIC ? (
          <div className="sim-status-callout">
            mission controller active - task behavior executing
          </div>
        ) : null}

        <CompositeScenePanel
          key={env.id}
          envId={env.id}
          missionName={env.name}
          policyEnabled={controllerActive}
        />
      </main>
    )
  }

  if (route.view === 'gym') {
    const env = getScenarioById(route.envId) ?? scenarios[0]

    return (
      <main className="app-shell app-shell--gym-full">
        <div className="gym-top-bar">
          <button type="button" className="sim-back-link" onClick={enterMenu}>
            Back to gym menu
          </button>
          <div className="sim-title-block">
            <span>Training Gym</span>
            <strong>{env.name}</strong>
          </div>
          {scopedNav(env.id, 'gym')}
          <div
            className={`sim-policy-badge ${
              policyStore[env.id] === 'ready' ? 'sim-policy-badge--ready' : ''
            }`}
          >
            <span className="sim-policy-badge__kicker">Policy</span>
            <span>{policyStore[env.id] ?? 'not-trained'}</span>
          </div>
        </div>

        <GymScenarioStage
          key={env.id}
          scenario={env}
          onPolicyReady={handlePolicyReady}
          onTrainingStart={handleTrainingStart}
          onTrainingError={handleTrainingError}
          policyStatus={policyStore[env.id] ?? 'not-trained'}
        />
      </main>
    )
  }

  return (
    <main className="app-shell app-shell--menu">
      <section className="menu-viewport">
        <div className="menu-header">
          <div className="menu-header-left">
            <span className="menu-subtitle">Select training environment</span>
            <h1 className="menu-title">Training Gym</h1>
            <p className="menu-kicker">
              Pick a scenario, train the controller, then open Mission Sim with
              the same environment and policy context.
            </p>
          </div>
          <button
            type="button"
            className="menu-action"
            onClick={enterCombatOS}
          >
            CombatOS
          </button>
        </div>

        <div className="menu-grid">
          {scenarios.map((scenario) => {
            const status = policyStore[scenario.id] ?? 'not-trained'
            const ready = status === 'ready'

            return (
              <article
                key={scenario.id}
                className={`menu-card ${ready ? 'menu-card--ready' : ''}`}
              >
                <div className="menu-card-preview">
                  <ScenarioMiniPreview scenarioId={scenario.id} />
                </div>
                <div className="menu-card-topline">
                  <div>
                    <span className="menu-card-meta-label">
                      {scenario.label}
                    </span>
                    <h2 className="menu-card-name">{scenario.name}</h2>
                  </div>
                  <span
                    className="menu-card-badge"
                    data-status={ready ? 'Ready' : 'Not trained'}
                  >
                    {ready ? 'Policy ready' : status}
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
                <div className="menu-card-actions">
                  <button
                    type="button"
                    className="menu-card-cta menu-card-cta--button"
                    onClick={() => enterGym(scenario.id)}
                  >
                    Train
                  </button>
                  <button
                    type="button"
                    className="menu-card-cta menu-card-cta--button"
                    onClick={() => enterSim(scenario.id)}
                    disabled={!simAllowedFor(scenario.id)}
                  >
                    Mission Sim
                  </button>
                </div>
              </article>
            )
          })}
        </div>
      </section>
      {toast ? <div className="app-toast">{toast}</div> : null}
    </main>
  )
}
