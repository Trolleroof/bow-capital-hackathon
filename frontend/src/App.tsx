import { useEffect, useState } from 'react'
import './App.css'
import CompositeScenePanel from './panels/CompositeScenePanel'
import GymScenarioStage from './gym/GymScenarioStage'
import { getScenarioById, scenarios } from './gym/scenarios'

type View = 'sim' | 'gym'

function getInitialView(): View {
  return window.location.hash === '#gym' ? 'gym' : 'sim'
}

function App() {
  const [view, setView] = useState<View>(getInitialView)
  const [selectedGymEnvironment, setSelectedGymEnvironment] = useState(scenarios[0].id)

  useEffect(() => {
    const onHashChange = () => setView(getInitialView())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const switchView = (nextView: View) => {
    window.location.hash = nextView === 'gym' ? 'gym' : 'sim'
    setView(nextView)
  }

  const gymEnvironment = getScenarioById(selectedGymEnvironment)

  const nav = (
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
  )

  return (
    <main className={`app-shell ${view === 'sim' ? 'app-shell--sim' : ''}`}>
      <div className="app-backdrop" />

      {view === 'sim' ? (
        <section className="sim-viewport-full" aria-label="Mission simulation">
          <div className="sim-viewport-nav">{nav}</div>
          <CompositeScenePanel
            missionName="Land Coverage Survey"
            missionBrief="Field reconstruction and coverage sweep"
          />
        </section>
      ) : (
        <>
          <header className="app-header">
            <div className="brand-lockup">
              <p className="eyebrow">BOW CAPITAL DRONE OPS</p>
              <h1>Cleaner mission control for sim and environment bring-up</h1>
              <span className="deck">
                Live coverage sim on one side, gym environment staging on the other.
              </span>
            </div>
            {nav}
          </header>
        <section className="gym-layout">
          <aside className="gym-sidebar panel-frame">
            <div className="panel-kicker">Environment registry</div>
            <h2>Gym page</h2>
            <p>
              Hard-coded battle drills live here now. Each scenario maps to an operator task, has
              visible telemetry, and points at a matching policy-training hook in `swarm/`.
            </p>

            <div className="gym-card-stack" role="list" aria-label="Selectable environments">
              {scenarios.map((environment) => {
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

            <GymScenarioStage key={gymEnvironment.id} scenario={gymEnvironment} />

            <div className="gym-notes" role="list" aria-label="Gym environment notes">
              <article role="listitem">
                <strong>Observation</strong>
                <span>{gymEnvironment.observation}</span>
              </article>
              <article role="listitem">
                <strong>Action + reward</strong>
                <span>
                  {gymEnvironment.action} Reward sketch: {gymEnvironment.reward}
                </span>
              </article>
              <article role="listitem">
                <strong>Scenario notes</strong>
                <span>{gymEnvironment.notes.join(' ')}</span>
              </article>
            </div>
          </section>
        </section>
        </>
      )}
    </main>
  )
}

export default App
