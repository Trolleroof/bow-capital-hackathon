import { useState, useEffect, useRef } from 'react'
import { useCombatState } from './useCombatState'
import { BootView } from './BootView'
import { CommandView } from './CommandView'
import { OpticView } from './OpticView'
import './combatos.css'

type View = 'boot' | 'command' | 'optic'

export function CombatOS() {
  const { t, log, followTarget, confirmTarget, releaseTarget } = useCombatState()
  const [view, setView] = useState<View>('boot')
  const [booted, setBooted] = useState(false)
  const canvasRef = useRef<HTMLDivElement>(null)

  const launch = () => { setBooted(true); setView('command') }
  const enterOptic = () => setView('optic')
  const exitOptic = () => setView('command')

  // keyboard shortcuts
  useEffect(() => {
    if (view === 'boot') return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') exitOptic()
      else if (e.key.toLowerCase() === 'o') setView(v => v === 'command' ? 'optic' : 'command')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [view])

  // scale 1440×900 canvas to fit viewport
  useEffect(() => {
    const fit = () => {
      const stage = document.getElementById('cos-stage-el')
      const cv = canvasRef.current
      if (!stage || !cv) return
      const W = stage.clientWidth, H = stage.clientHeight
      const sc = Math.min(W / 1440, H / 900)
      cv.style.transform = `translate(${(W - 1440 * sc) / 2}px, ${(H - 900 * sc) / 2}px) scale(${sc})`
    }
    fit()
    const id = setTimeout(fit, 60)
    window.addEventListener('resize', fit)
    return () => { clearTimeout(id); window.removeEventListener('resize', fit) }
  }, [booted])

  return (
    <div className="cos-stage" id="cos-stage-el">
      <div className="cos-canvas" ref={canvasRef}>
        {booted && (
          <>
            <div className={'cos-layer cos-layer--cmd' + (view === 'optic' ? ' is-dim' : '')}>
              <CommandView t={t} log={log} onEnterOptic={enterOptic} onConfirm={confirmTarget} />
            </div>
            <div className={'cos-layer cos-layer--optic' + (view === 'optic' ? ' is-show' : '')}>
              <OpticView t={t} onExit={exitOptic} onFollow={followTarget} onConfirm={confirmTarget} onRelease={releaseTarget} />
            </div>
          </>
        )}
      </div>

      {view === 'boot' && <BootView onLaunch={launch} />}
    </div>
  )
}
