/**
 * BattlefieldParamsPanel.tsx
 *
 * Issue #25 — Slim panel for P0 battlefield knobs.
 *
 * • Preset toggle: Garrison (low-stress baseline) vs Combat (per-scenario defaults).
 * • Exposes only P0 knobs that wire directly into sim dynamics / obs / rewards.
 * • Validates on every change via validateBattlefieldParams.
 * • Disabled during training so params stay locked to the running job.
 */

import type { BattlefieldParams } from './battlefieldParams'
import {
  PARAM_BOUNDS,
  getScenarioDefaults,
  validateBattlefieldParams,
} from './battlefieldParams'

// ─────────────────────────────── garrison baseline (mirror of env_config.py) ──

function getGarrisonDefaults(envId: string): BattlefieldParams {
  return {
    envId,
    weather:  { windSpeed: 0, windDirRad: 0, visibility: 1.0, temperatureC: 20 },
    ew:       { gpsDenialLevel: 0, jamDutyCycle: 0, spoofingEnabled: false },
    terrain:  { elevRoughness: 0, urbanDensity: 0 },
    threat:   { hostileUasCount: 0, movingTargetSpeed: 0.3 },
    roe:      { engagementAuthority: 'hold-fire', minStandoffM: 0, civilianDensity: 0, timeLimitSec: 400 },
    logistics:{ swarmSize: 5, batteryEnvelopeSec: 400, attritionInjectRate: 0 },
  }
}

function detectPreset(params: BattlefieldParams): 'garrison' | 'combat' {
  // Heuristic: garrison has zero attrition and no GPS denial
  return params.logistics.attritionInjectRate < 0.01 && params.ew.gpsDenialLevel < 0.05
    ? 'garrison'
    : 'combat'
}

// ────────────────────────────── P0 knob manifest ──────────────────────────

interface KnobDef {
  group: 'logistics' | 'ew' | 'weather' | 'roe'
  key: string
  label: string
  unit: string
  decimals: number
  step: number
}

const P0_KNOBS: KnobDef[] = [
  { group: 'logistics', key: 'swarmSize',           label: 'Swarm',      unit: '',   decimals: 0, step: 1    },
  { group: 'logistics', key: 'attritionInjectRate', label: 'Attrition',  unit: '',   decimals: 2, step: 0.01 },
  { group: 'logistics', key: 'batteryEnvelopeSec',  label: 'Battery',    unit: 's',  decimals: 0, step: 10   },
  { group: 'ew',        key: 'gpsDenialLevel',      label: 'GPS Denial', unit: '',   decimals: 2, step: 0.05 },
  { group: 'ew',        key: 'jamDutyCycle',        label: 'Jam Duty',   unit: '',   decimals: 2, step: 0.05 },
  { group: 'weather',   key: 'windSpeed',           label: 'Wind',       unit: 'm/s',decimals: 1, step: 0.5  },
  { group: 'roe',       key: 'timeLimitSec',        label: 'Time Limit', unit: 's',  decimals: 0, step: 10   },
]

// ─────────────────────────────────────────────────── component ─────────────

export interface BattlefieldParamsPanelProps {
  params: BattlefieldParams
  onChange: (next: BattlefieldParams) => void
  isTraining: boolean
  onTrain: () => void
  onStop: () => void
  open: boolean
  onToggleOpen: () => void
}

export default function BattlefieldParamsPanel({
  params,
  onChange,
  isTraining,
  onTrain,
  onStop,
  open,
  onToggleOpen,
}: BattlefieldParamsPanelProps) {
  const errors = validateBattlefieldParams(params)
  const preset = detectPreset(params)

  function applyPreset(p: 'garrison' | 'combat') {
    onChange(
      p === 'combat'
        ? getScenarioDefaults(params.envId)
        : getGarrisonDefaults(params.envId),
    )
  }

  function setKnob(group: KnobDef['group'], key: string, value: number) {
    onChange({
      ...params,
      [group]: { ...(params[group] as unknown as Record<string, unknown>), [key]: value },
    })
  }

  return (
    <>
      {/* ── compact toolbar row ───────────────────────────────────────── */}
      <div className="gym-training-bar">
        {/* preset toggle */}
        <div
          className="gym-preset-toggle"
          role="group"
          aria-label="Stress preset"
        >
          <button
            type="button"
            className={preset === 'garrison' ? 'is-active' : ''}
            onClick={() => applyPreset('garrison')}
            disabled={isTraining}
            title="Garrison: zero-stress baseline"
          >
            Garrison
          </button>
          <button
            type="button"
            className={preset === 'combat' ? 'is-active' : ''}
            onClick={() => applyPreset('combat')}
            disabled={isTraining}
            title="Combat: per-scenario stress defaults"
          >
            Combat
          </button>
        </div>

        {/* params toggle */}
        <button
          type="button"
          className={`gym-params-toggle ${open ? 'is-active' : ''}`}
          onClick={onToggleOpen}
          disabled={isTraining}
          aria-expanded={open}
          aria-controls="gym-params-panel"
          title="Toggle P0 parameter knobs"
        >
          Params {open ? '▲' : '▼'}
          {errors.length > 0 && (
            <span className="gym-params-error-badge" aria-label={`${errors.length} validation error`}>
              {errors.length}
            </span>
          )}
        </button>

        {/* train / stop */}
        <button
          type="button"
          className={`gym-train-btn ${isTraining ? 'gym-train-btn--stop' : ''}`}
          onClick={isTraining ? onStop : onTrain}
          disabled={!isTraining && errors.length > 0}
          aria-label={isTraining ? 'Stop training' : 'Start training policy'}
        >
          {isTraining ? '■ Stop Training' : '▶ Train Policy'}
        </button>
      </div>

      {/* ── collapsible P0 knobs grid ─────────────────────────────────── */}
      {open && !isTraining && (
        <div
          id="gym-params-panel"
          className="gym-params-panel"
          role="form"
          aria-label="P0 battlefield parameters"
        >
          {P0_KNOBS.map(({ group, key, label, unit, decimals, step }) => {
            const bounds = (PARAM_BOUNDS as Record<string, Record<string, { min: number; max: number }>>)[group]?.[key]
            if (!bounds) return null

            const value = ((params[group] as unknown as Record<string, number>)[key]) as number
            const displayValue = decimals > 0 ? value.toFixed(decimals) : String(value)
            const inputId = `knob-${group}-${key}`

            return (
              <div key={inputId} className="gym-knob">
                <label htmlFor={inputId}>{label}</label>
                <div className="gym-knob-row">
                  <input
                    id={inputId}
                    type="range"
                    min={bounds.min}
                    max={bounds.max}
                    step={step}
                    value={value}
                    onChange={e => setKnob(group, key, parseFloat(e.target.value))}
                  />
                  <span className="gym-knob-value" aria-label={`${label} value`}>
                    {displayValue}{unit}
                  </span>
                </div>
              </div>
            )
          })}

          {/* validation errors */}
          {errors.length > 0 && (
            <div className="gym-params-errors" role="alert">
              {errors.map(e => (
                <p key={e.field} className="gym-params-error">
                  {e.message}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  )
}
