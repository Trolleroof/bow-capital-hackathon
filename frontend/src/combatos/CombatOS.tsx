import { useState, useEffect } from 'react'
import { useCombatState } from './useCombatState'
import { BootView } from './BootView'
import { CommandView } from './CommandView'
import { ExpandedPanelView } from './ExpandedPanelView'
import { OpticView } from './OpticView'
import type { CommandPanelId } from './panels'
import './combatos.css'

type View = 'boot' | 'command' | 'optic'

export function CombatOS() {
  const { t, log, followTarget, confirmTarget, releaseTarget } = useCombatState()
  const [view, setView] = useState<View>('boot')
  const [booted, setBooted] = useState(false)
  const [expandedPanel, setExpandedPanel] = useState<CommandPanelId | null>(null)

  const launch = () => { setBooted(true); setView('command') }
  const enterOptic = () => { setExpandedPanel(null); setView('optic') }
  const exitOptic = () => setView('command')
  const expandPanel = (panel: CommandPanelId) => setExpandedPanel(panel)
  const collapsePanel = () => setExpandedPanel(null)

  useEffect(() => {
    if (view === 'boot') return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (view === 'optic') exitOptic()
        else if (expandedPanel) collapsePanel()
      } else if (e.key.toLowerCase() === 'o' && view === 'command' && !expandedPanel) {
        setView(v => v === 'command' ? 'optic' : 'command')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [view, expandedPanel])

  const overlayActive = view === 'optic' || expandedPanel != null

  return (
    <div className="cos-stage" id="cos-stage-el">
      {view === 'boot' && <BootView onLaunch={launch} />}
      {booted && (
        <div className="cos-canvas">
          <div className={'cos-layer cos-layer--cmd' + (overlayActive ? ' is-dim' : '')}>
            <CommandView
              t={t}
              log={log}
              onEnterOptic={enterOptic}
              onConfirm={confirmTarget}
              onExpandPanel={expandPanel}
            />
          </div>
          <div className={'cos-layer cos-layer--expanded' + (expandedPanel ? ' is-show' : '')}>
            {expandedPanel && (
              <ExpandedPanelView
                panel={expandedPanel}
                t={t}
                onExit={collapsePanel}
                onConfirm={confirmTarget}
                onEnterOptic={enterOptic}
              />
            )}
          </div>
          <div className={'cos-layer cos-layer--optic' + (view === 'optic' ? ' is-show' : '')}>
            <OpticView t={t} onExit={exitOptic} onFollow={followTarget} onConfirm={confirmTarget} onRelease={releaseTarget} />
          </div>
        </div>
      )}
    </div>
  )
}
