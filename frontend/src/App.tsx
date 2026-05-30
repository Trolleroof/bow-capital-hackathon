import SwarmPanel from './panels/SwarmPanel'

function App() {
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: '#06090c',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'monospace',
      }}
    >
      <header
        style={{
          padding: '10px 16px',
          color: '#4ef0a0',
          borderBottom: '1px solid #16313a',
          letterSpacing: 2,
          fontWeight: 700,
        }}
      >
        COMBATOS · SWARM
        <span style={{ opacity: 0.6, fontWeight: 400, marginLeft: 12 }}>
          GPS: DENIED · LINK: NONE · decentralized MAPPO policy
        </span>
      </header>
      <div style={{ flex: 1, minHeight: 0 }}>
        <SwarmPanel />
      </div>
    </div>
  )
}

export default App
