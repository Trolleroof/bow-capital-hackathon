import { useEffect, useState } from 'react'
import './App.css'
import CompositeScenePanel from './panels/CompositeScenePanel'

type View = 'sim' | 'gym'
type EnvironmentStatus = 'Ready' | 'Queued'

interface EnvironmentCard {
  id: string
  name: string
  label: string
  status: EnvironmentStatus
}

const environments: EnvironmentCard[] = [
  {
    id: 'land-coverage',
    name: 'Land Coverage Survey',
    label: 'Live now',
    status: 'Ready',
  },
  {
    id: 'gym-floor',
    name: 'Gym Floor',
    label: 'Hard-coded',
    status: 'Ready',
  },
  {
    id: 'warehouse',
    name: 'Warehouse Lanes',
    label: 'Stub',
    status: 'Queued',
  },
  {
    id: 'canyon',
    name: 'Canyon Corridor',
    label: 'Stub',
    status: 'Queued',
  },
]

function getInitialView(): View {
  return window.location.hash === '#gym' ? 'gym' : 'sim'
}

function App() {
  const [view, setView] = useState<View>(getInitialView)
  const [selectedGymEnvironment, setSelectedGymEnvironment] = useState('gym-floor')
  const isSimView = view === 'sim'

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
    <main className={`app-shell ${isSimView ? 'app-shell--sim' : ''}`}>
      <div className="app-backdrop" />

      <header className={`app-header ${isSimView ? 'app-header--compact' : ''}`}>
        {!isSimView ? <h1 className="app-title">Mission Control</h1> : null}
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

      {isSimView ? (
        <section className="sim-layout">
          <section className="sim-panel panel-frame">
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
            <h2>Gym environments</h2>

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
                <p>Selected environment</p>
                <h3>{gymEnvironment.name}</h3>
                <span>{gymEnvironment.status}</span>
              </div>
            </div>
          </section>
        </section>
      )}
    </main>
  )
}

export default App
