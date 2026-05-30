import './App.css'
import CompositeScenePanel from './panels/CompositeScenePanel'

function App() {
  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p>COMBATOS</p>
          <h1>Swarm through reconstructed field</h1>
        </div>
        <span>3DGS mock · VSLAM mock · MAPPO swarm</span>
      </header>
      <CompositeScenePanel />
    </main>
  )
}

export default App
