import { useEffect, useState } from 'react'
import './App.css'
import CompositeScenePanel from './panels/CompositeScenePanel'
import GymScenarioStage from './gym/GymScenarioStage'
import { getScenarioById, scenarios } from './gym/scenarios'
import { CombatOS } from './combatos/CombatOS'

type View = 'sim' | 'gym' | 'combatos'

function getInitialView(): View {
  if (window.location.hash === '#gym') return 'gym'
  if (window.location.hash === '#combatos') return 'combatos'
  return 'sim'
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
    window.location.hash = nextView === 'gym' ? 'gym' : nextView === 'combatos' ? 'combatos' : 'sim'
    setView(nextView)
  }

  const gymEnvironment = getScenarioById(selectedGymEnvironment)

  if (view === 'combatos') {
    return <CombatOS />
  }

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
      <button
        type="button"
        onClick={() => switchView('combatos')}
      >
        CombatOS
      </button>
    </nav>
  )

  return (
    <main className={`app-shell ${view === 'sim' ? 'app-shell--sim' : 'app-shell--gym'}`}>
      <div className="app-backdrop" />

      {view === 'sim' ? (
        <section className="sim-viewport-full" aria-label="Mission simulation">
          <div className="sim-viewport-nav">{nav}</div>
          <CompositeScenePanel missionName="Land Coverage Survey" />
        </section>
      ) : (
        <div className="gym-viewport">
          <header className="app-header app-header--minimal">
            {nav}
          </header>
          <section className="gym-layout">
            <aside className="gym-sidebar panel-frame">
              <div className="panel-kicker">Environments</div>

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
                    </button>
                  )
                })}
              </div>
            </aside>

            <section className="gym-stage panel-frame">
              <div className="panel-head">
                <h2>{gymEnvironment.name}</h2>
                <span data-status={gymEnvironment.status}>{gymEnvironment.status}</span>
              </div>

              <GymScenarioStage key={gymEnvironment.id} scenario={gymEnvironment} />
            </section>
          </section>
        </div>
      )}
    </main>
  )
}

export default App
