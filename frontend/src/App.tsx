import { useEffect, useRef, useState } from 'react'
import './App.css'
import CompositeScenePanel from './panels/CompositeScenePanel'
import GymScenarioStage from './gym/GymScenarioStage'
import { getScenarioById, scenarios } from './gym/scenarios'
import { CombatOS } from './combatos/CombatOS'
import {
  checkPolicyExists,
  setActiveEnvId,
  type PolicyStatus,
} from './swarm/policy'

// Dev escape: VITE_ALLOW_HEURISTIC_MISSION=1 skips policy gate in Mission Sim.
const ALLOW_HEURISTIC = import.meta.env.VITE_ALLOW_HEURISTIC_MISSION === '1'

// ─── Routing ─────────────────────────────────────────────────────────────────

type AppRoute =
  | { view: 'gym-registry' }
  | { view: 'gym-env'; envId: string }
  | { view: 'sim' }
  | { view: 'combatos' }

function parseRoute(): AppRoute {
  const raw = window.location.hash.replace(/^#/, '')
  if (!raw || raw === 'gym') return { view: 'gym-registry' }
  if (raw.startsWith('gym/')) {
    const envId = raw.slice(4)
    return { view: 'gym-env', envId: envId || scenarios[0].id }
  }
  if (raw === 'sim') return { view: 'sim' }
  if (raw === 'combatos') return { view: 'combatos' }
  return { view: 'gym-registry' }
}

function pushHash(route: AppRoute) {
  if (route.view === 'gym-registry') window.location.hash = 'gym'
  else if (route.view === 'gym-env') window.location.hash = `gym/${route.envId}`
  else if (route.view === 'combatos') window.location.hash = 'combatos'
  else window.location.hash = 'sim'
}

// ─── App ─────────────────────────────────────────────────────────────────────

function App() {
  const [route, setRoute] = useState<AppRoute>(parseRoute)

  // Last env selected / entered in the gym — drives registry preview & sim.
  const [selectedEnvId, setSelectedEnvId] = useState(scenarios[0].id)

  // The env whose checkpoint is (or will be) loaded into Mission Sim.
  const [simEnvId, setSimEnvId] = useState(scenarios[0].id)

  const [policyStore, setPolicyStore] = useState<Record<string, PolicyStatus>>(
    () =>
      Object.fromEntries(
        scenarios.map((s) => [s.id, 'not-trained' as PolicyStatus]),
      ),
  )

  const [toast, setToast] = useState<string | null>(null)
  const toastTimer = useRef<number | null>(null)

  // ── Hash-driven routing ──────────────────────────────────────────────────
  useEffect(() => {
    // Fresh load always lands on #gym, not sim.
    if (!window.location.hash || window.location.hash === '#') {
      window.location.hash = 'gym'
    }
    const onHashChange = () => setRoute(parseRoute())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  // ── Probe checkpoints on mount ───────────────────────────────────────────
  useEffect(() => {
    scenarios.forEach(async (s) => {
      const exists = await checkPolicyExists(s.id)
      if (exists) {
        setPolicyStore((prev) => ({ ...prev, [s.id]: 'ready' }))
      }
    })
  }, [])

  // ── Derived: is Mission Sim reachable? ───────────────────────────────────
  const hasAnyReadyPolicy = Object.values(policyStore).some(
    (s) => s === 'ready',
  )
  const simAllowed = hasAnyReadyPolicy || ALLOW_HEURISTIC

  // ── Guard: redirect sim → gym if policy gate blocks access ───────────────
  useEffect(() => {
    if (route.view === 'sim' && !simAllowed) {
      showToast('Train a policy in Gym first.')
      setRoute({ view: 'gym-registry' })
      window.location.hash = 'gym'
    }
  }, [route.view, simAllowed])

  // ── Toast ────────────────────────────────────────────────────────────────
  function showToast(msg: string) {
    setToast(msg)
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast(null), 4000)
  }

  const switchView = (next: 'combatos' | 'sim' | 'gym') => {
    if (next === 'combatos') {
      window.location.hash = 'combatos'
      setRoute({ view: 'combatos' })
    } else if (next === 'gym') {
      window.location.hash = 'gym'
      setRoute({ view: 'gym-registry' })
    } else {
      goToSim()
    }
  }

  // ── Navigation ───────────────────────────────────────────────────────────

  if (route.view === 'combatos') {
    return <CombatOS />
  }

  const goToRegistry = () => {
    const next: AppRoute = { view: 'gym-registry' }
    pushHash(next)
    setRoute(next)
  }

  const enterGymEnv = (envId: string) => {
    setSelectedEnvId(envId)
    const next: AppRoute = { view: 'gym-env', envId }
    pushHash(next)
    setRoute(next)
  }

  const goToSim = () => {
    if (!simAllowed) {
      showToast('Train a policy in Gym first.')
      return
    }

    const activeEnvId =
      route.view === 'gym-env' ? route.envId : selectedEnvId
    const chosenEnvId =
      policyStore[activeEnvId] === 'ready' || ALLOW_HEURISTIC
        ? activeEnvId
        : (scenarios.find((s) => policyStore[s.id] === 'ready')?.id ??
          activeEnvId)

    setActiveEnvId(chosenEnvId)
    setSimEnvId(chosenEnvId)
    setSelectedEnvId(chosenEnvId)

    const next: AppRoute = { view: 'sim' }
    pushHash(next)
    setRoute(next)
  }

  const goToCombatOS = () => {
    const next: AppRoute = { view: 'combatos' }
    pushHash(next)
    setRoute(next)
  }

  // ── Render: CombatOS (full takeover, no app-shell chrome) ────────────────
  if (route.view === 'combatos') {
    return <CombatOS />
  }

  // ── Derived ──────────────────────────────────────────────────────────────
  const gymFullEnvId =
    route.view === 'gym-env' ? route.envId : selectedEnvId
  const gymEnv = getScenarioById(gymFullEnvId)
  const registryEnv = getScenarioById(selectedEnvId)

  // ── Shared top-nav ───────────────────────────────────────────────────────
  const nav = (
    <nav className="top-nav" aria-label="Primary">
      <button
        type="button"
        className={route.view === 'sim' ? 'is-active' : undefined}
        onClick={goToSim}
        disabled={!simAllowed}
        title={!simAllowed ? 'Train a policy in Gym first.' : undefined}
      >
        Mission Sim
      </button>
      <button
        type="button"
        className={route.view !== 'sim' && route.view !== 'combatos' ? 'is-active' : undefined}
        onClick={goToRegistry}
      >
        Gym Environments
      </button>
      <button
        type="button"
        onClick={goToCombatOS}
      >
        CombatOS
      </button>
    </nav>
  )

  // ── Render: fullscreen gym environment ───────────────────────────────────
  if (route.view === 'gym-env') {
    return (
      <main className="app-shell app-shell--gym-full">
        <div className="app-backdrop" />
        <section
          className="gym-scene-full"
          aria-label={`${gymEnv.name} training environment`}
        >
          <div className="gym-scene-full-nav">
            <button
              type="button"
              className="gym-back-btn"
              onClick={goToRegistry}
              aria-label="Back to environment registry"
            >
              ← Environments
            </button>
            <div className="gym-scene-full-meta">
              <span className="gym-scene-full-name">{gymEnv.name}</span>
              <em
                data-status={
                  policyStore[gymEnv.id] === 'ready' ? 'Ready' : gymEnv.status
                }
              >
                {policyStore[gymEnv.id] === 'ready'
                  ? 'Policy ready'
                  : gymEnv.label}
              </em>
            </div>
            <div className="gym-scene-full-right">{nav}</div>
          </div>

          <GymScenarioStage key={gymEnv.id} scenario={gymEnv} />
        </section>

        {toast && (
          <div className="app-toast" role="status">
            {toast}
          </div>
        )}
      </main>
    )
  }

  // ── Render: Mission Sim ──────────────────────────────────────────────────
  if (route.view === 'sim' && simAllowed) {
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
              {policyReady ? 'ONNX' : 'Heuristic'}
            </em>
          </div>

          <CompositeScenePanel key={simEnvId} missionName={simEnv.name} />
        </section>

        {toast && (
          <div className="app-toast" role="status">
            {toast}
          </div>
        )}
      </main>
    )
  }

  // ── Render: Gym registry (default / #gym) ────────────────────────────────
  return (
    <main className="app-shell app-shell--gym">
      <div className="app-backdrop" />

      <div className="gym-viewport">
        <header className="app-header app-header--minimal">{nav}</header>

        <section className="gym-layout">
          <aside className="gym-sidebar panel-frame">
            <div className="panel-kicker">Environments</div>

            <div
              className="gym-card-stack"
              role="list"
              aria-label="Selectable environments"
            >
              {scenarios.map((env) => {
                const isSelected = selectedEnvId === env.id
                const status = policyStore[env.id]
                return (
                  <button
                    key={env.id}
                    type="button"
                    className={`gym-card ${isSelected ? 'is-selected' : ''}`}
                    onClick={() => enterGymEnv(env.id)}
                    aria-label={`Enter ${env.name} fullscreen`}
                  >
                    <span className="gym-card-topline">
                      <strong>{env.name}</strong>
                      <em
                        data-status={
                          status === 'ready'
                            ? 'Ready'
                            : status === 'training'
                              ? 'Queued'
                              : env.status
                        }
                      >
                        {status === 'ready'
                          ? 'Policy ready'
                          : status === 'training'
                            ? 'Training…'
                            : env.status}
                      </em>
                    </span>
                    <span className="gym-card-label">{env.label}</span>
                  </button>
                )
              })}
            </div>
          </aside>

          <section className="gym-stage panel-frame">
            <div className="panel-head">
              <h2>{registryEnv.name}</h2>
              <span
                data-status={
                  policyStore[registryEnv.id] === 'ready'
                    ? 'Ready'
                    : registryEnv.status
                }
              >
                {policyStore[registryEnv.id] === 'ready'
                  ? 'Policy ready'
                  : registryEnv.status}
              </span>
            </div>

            <GymScenarioStage key={registryEnv.id} scenario={registryEnv} />
          </section>
        </section>
      </div>

      {toast && (
        <div className="app-toast" role="status">
          {toast}
        </div>
      )}
    </main>
  )
}

export default App
