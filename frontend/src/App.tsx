import { useEffect, useRef, useState } from 'react'
import './App.css'
import PyBulletSimPanel from './panels/PyBulletSimPanel'
import GymScenarioStage, { ScenarioMiniPreview } from './gym/GymScenarioStage'
import { getScenarioById, scenarios } from './gym/scenarios'
import { startPyBulletSim } from './gym/trainApi'
import { OutcastVirus } from './outcast-virus/OutcastVirus'
import {
  checkPolicyExists,
  setActiveEnvId,
  type PolicyStatus,
} from './swarm/policy'

const ALLOW_HEURISTIC = false

type AppRoute =
  | { view: 'menu' }
  | { view: 'gym'; envId: string }
  | { view: 'sim'; envId: string }
  | { view: 'outcast-virus' }

const scenarioExists = (envId: string) => getScenarioById(envId) != null

const parseRoute = (): AppRoute => {
  const hash = window.location.hash.replace(/^#\/?/, '')
  if (!hash) return { view: 'outcast-virus' }

  const [view, envId] = hash.split('/')

  if (view === 'outcast-virus') return { view: 'outcast-virus' }

  if (view === 'menu') return { view: 'menu' }

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
  const [launchingSim, setLaunchingSim] = useState<string | null>(null)
  const toastTimer = useRef<number | null>(null)

  useEffect(() => {
    if (!window.location.hash) {
      setHashRoute('outcast-virus')
      setRoute({ view: 'outcast-virus' })
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

  const enterSim = async (envId: string) => {
    if (!simAllowedFor(envId)) {
      showToast('Train this environment first to unlock PyBullet Sim.')
      return
    }

    setLaunchingSim(envId)
    const result = await startPyBulletSim(envId)
    setLaunchingSim((current) => (current === envId ? null : current))

    if (!result.ok) {
      showToast(result.error ?? 'PyBullet Sim failed to start.')
      return
    }

    setActiveEnvId(envId)
    setHashRoute(`sim/${envId}`)
    setRoute({ view: 'sim', envId })
  }

  const activateScenarioCard = (envId: string) => {
    enterGym(envId)
  }

  const enterOutcastVirus = () => {
    setHashRoute('outcast-virus')
    setRoute({ view: 'outcast-virus' })
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
          onClick={() => void enterSim(envId)}
          disabled={simLocked || launchingSim === envId}
          title={simLocked ? 'Train this environment first.' : undefined}
        >
          {launchingSim === envId ? 'Launching' : 'PyBullet Sim'}
        </button>
        <button type="button" className="nav-pill" onClick={enterOutcastVirus}>
          Outcast Virus
        </button>
      </div>
    )
  }

  if (route.view === 'outcast-virus') {
    return <OutcastVirus />
  }

  if (route.view === 'sim') {
    const env = getScenarioById(route.envId) ?? scenarios[0]
    const policyReady = policyStore[env.id] === 'ready'

    return (
      <main className="app-shell app-shell--sim">
        <div className="sim-top-bar">
          <button type="button" className="sim-back-link" onClick={enterMenu}>
            Back to gym menu
          </button>
          <div className="sim-title-block">
            <span>PyBullet Sim</span>
            <strong>{env.name}</strong>
          </div>
          {scopedNav(env.id, 'sim')}
          <div
            className={`sim-policy-badge ${
              policyReady ? 'sim-policy-badge--ready' : ''
            }`}
          >
            <span className="sim-policy-badge__kicker">Controller</span>
            <span>
              {policyReady ? 'Trained policy' : 'Locked'}
            </span>
          </div>
        </div>

        <PyBulletSimPanel
          key={env.id}
          envId={env.id}
          missionName={`${env.name} PyBullet Sim`}
        />
      </main>
    )
  }

  if (route.view === 'gym') {
    const env = getScenarioById(route.envId) ?? scenarios[0]
    const policyReady = policyStore[env.id] === 'ready'

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
          canLaunchSim={policyReady}
          onLaunchSim={() => void enterSim(env.id)}
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
            <h1 className="menu-title">Training Gym</h1>

          </div>
          <button
            type="button"
            className="menu-action"
            onClick={enterOutcastVirus}
          >
            Outcast Virus
          </button>
        </div>

        <div className="menu-grid">
          {scenarios.map((scenario) => (
              <article
                key={scenario.id}
                className="menu-card"
                role="button"
                tabIndex={0}
                onClick={() => activateScenarioCard(scenario.id)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    activateScenarioCard(scenario.id)
                  }
                }}
              >
                <div className="menu-card-preview">
                  <ScenarioMiniPreview scenarioId={scenario.id} />
                </div>
                <div className="menu-card-topline">
                  <span className="menu-card-meta-label">{scenario.label}</span>
                  <h2 className="menu-card-name">{scenario.name}</h2>
                </div>
                <p className="menu-card-summary">{scenario.summary}</p>
              </article>
          ))}
        </div>
      </section>
      {toast ? <div className="app-toast">{toast}</div> : null}
    </main>
  )
}
