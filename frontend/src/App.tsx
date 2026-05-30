import { useEffect, useState } from 'react'
import './App.css'
import CompositeScenePanel from './panels/CompositeScenePanel'

type View = 'sim' | 'gym'
type EnvironmentStatus = 'Ready' | 'Queued'

interface EnvironmentCard {
  id: string
  name: string
  label: string
  summary: string
  status: EnvironmentStatus
}

const environments: EnvironmentCard[] = [
  {
    id: 'land-coverage',
    name: 'Land Coverage Survey',
    label: 'Live now',
    summary: 'Current field-reconstruction mission with swarm coverage and local policy playback.',
    status: 'Ready',
  },
  {
    id: 'gym-floor',
    name: 'Gym Floor',
    label: 'Hard-coded',
    summary: 'Indoor calibration gym for staged environment bring-up. Blank shell for asset wiring.',
    status: 'Ready',
  },
  {
    id: 'warehouse',
    name: 'Warehouse Lanes',
    label: 'Stub',
    summary: 'Reserved slot for dense aisle navigation and obstacle choreography.',
    status: 'Queued',
  },
  {
    id: 'canyon',
    name: 'Canyon Corridor',
    label: 'Stub',
    summary: 'Reserved slot for constrained line-of-sight and terrain-follow behavior.',
    status: 'Queued',
  },
]

function getInitialView(): View {
  return window.location.hash === '#gym' ? 'gym' : 'sim'
}

function App() {
  const [view, setView] = useState<View>(getInitialView)
  const [selectedGymEnvironment, setSelectedGymEnvironment] = useState('gym-floor')

  useEffect(() => {
    const onHashChange = () => setView(getInitialView())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const switchView = (nextView: View) => {
    window.location.hash = nextView === 'gym' ? 'gym' : 'sim'
    setView(nextView)
  }

  const gymEnvironment =
    environments.find((environment) => environment.id === selectedGymEnvironment) ?? environments[1]

  return (
    <main className="app-shell">
      <div className="app-backdrop" />

      <header className="app-header">
        <div className="brand-lockup">
          <p className="eyebrow">BOW CAPITAL DRONE OPS</p>
          <h1>Cleaner mission control for sim and environment bring-up</h1>
          <span className="deck">
            Live coverage sim on one side, gym environment staging on the other.
          </span>
        </div>

        <nav className="top-nav" aria-label="Primary">
          <button
            type="button"
            className={view === 'sim' ? 'is-active' : undefined}
            onClick={() => switchView('sim')}
          >
            Mission Sim
          </button>
          <button
            type="button"
            className={view === 'gym' ? 'is-active' : undefined}
            onClick={() => switchView('gym')}
          >
            Gym Environments
          </button>
        </nav>
      </header>

      {view === 'sim' ? (
        <section className="dashboard-grid">
          <aside className="mission-panel panel-frame">
            <div className="panel-kicker">Active mission</div>
            <h2>Land Coverage Survey</h2>
            <p>
              The existing environment is reframed as a coverage survey mission so the sim reads
              clearly: swarm health, coverage progress, denied-link posture, and reconstruction
              context are all visible without the old clutter.
            </p>

            <div className="metric-row" role="list" aria-label="Mission highlights">
              <article role="listitem">
                <strong>3D field</strong>
                <span>Reconstruction backdrop and flight path trace stay live in the scene.</span>
              </article>
              <article role="listitem">
                <strong>Coverage-first</strong>
                <span>The environment communicates survey progress instead of vague land-use text.</span>
              </article>
              <article role="listitem">
                <strong>Operator-ready</strong>
                <span>Reset and revive controls remain accessible inside the sim card.</span>
              </article>
            </div>

            <div className="environment-list" role="list" aria-label="Environment registry">
              {environments.map((environment) => (
                <article className="environment-row" key={environment.id} role="listitem">
                  <div>
                    <p>{environment.name}</p>
                    <span>{environment.summary}</span>
                  </div>
                  <strong data-status={environment.status}>{environment.label}</strong>
                </article>
              ))}
            </div>
          </aside>

          <section className="sim-panel panel-frame">
            <div className="panel-head">
              <div>
                <p className="panel-kicker">Live scene</p>
                <h2>Field reconstruction viewport</h2>
              </div>
              <span>Local policy • orbit camera • coverage telemetry</span>
            </div>
            <CompositeScenePanel
              missionName="Land Coverage Survey"
              missionBrief="Field reconstruction and coverage sweep"
            />
          </section>
        </section>
      ) : (
        <section className="gym-layout">
          <aside className="gym-sidebar panel-frame">
            <div className="panel-kicker">Environment registry</div>
            <h2>Gym page</h2>
            <p>
              This page is the staging area for hard-coded environments. Gym Floor is wired in now,
              and the remaining environments are left blank on purpose so assets and rules can be
              filled in next.
            </p>

            <div className="gym-card-stack" role="list" aria-label="Selectable environments">
              {environments.map((environment) => {
                const isSelected = selectedGymEnvironment === environment.id
                return (
                  <button
                    key={environment.id}
                    type="button"
                    className={`gym-card ${isSelected ? 'is-selected' : ''}`}
                    onClick={() => setSelectedGymEnvironment(environment.id)}
                  >
                    <span className="gym-card-topline">
                      <strong>{environment.name}</strong>
                      <em data-status={environment.status}>{environment.status}</em>
                    </span>
                    <span className="gym-card-label">{environment.label}</span>
                    <span className="gym-card-summary">{environment.summary}</span>
                  </button>
                )
              })}
            </div>
          </aside>

          <section className="gym-stage panel-frame">
            <div className="panel-head">
              <div>
                <p className="panel-kicker">Selected environment</p>
                <h2>{gymEnvironment.name}</h2>
              </div>
              <span>{gymEnvironment.label}</span>
            </div>

            <div className="blank-stage" aria-label={`${gymEnvironment.name} placeholder`}>
              <div className="blank-stage-grid" />
              <div className="blank-stage-content">
                <p>Blank environment shell</p>
                <h3>{gymEnvironment.name}</h3>
                <span>{gymEnvironment.summary}</span>
              </div>
            </div>

            <div className="gym-notes" role="list" aria-label="Gym environment notes">
              <article role="listitem">
                <strong>Gym Floor is hard-coded</strong>
                <span>Use this slot for the first indoor environment pass and object layout work.</span>
              </article>
              <article role="listitem">
                <strong>Other environments stay blank</strong>
                <span>They are visible in the registry so the navigation model is in place now.</span>
              </article>
              <article role="listitem">
                <strong>Next step ready</strong>
                <span>Additional hard-coded environments can drop into this page without changing app structure.</span>
              </article>
            </div>
          </section>
        </section>
      )}
    </main>
  )
}

export default App
