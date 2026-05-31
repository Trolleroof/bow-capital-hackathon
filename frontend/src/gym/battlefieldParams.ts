/**
 * battlefieldParams.ts — Typed schema for CombatOS battlefield simulation parameters.
 *
 * Priority tiers (see docs/battlefield-parameters.md §2):
 *   P0  — wired into sim dynamics / obs / rewards (swarm/env.py + sim.ts)
 *   P1  — UI display only; no sim change at hackathon scope (annotated below)
 *   P2  — deferred post-demo
 *
 * Mirror: swarm/env_config.py  (Python dataclass, same field names / ranges)
 */

// ─────────────────────────────────────────────────────────────── sub-types ──

/** P0: wind drift; P1: visibility and temperature */
export interface WeatherParams {
  /** [P0] Wind speed in world-units/s (0–15).  Affects position integration. */
  windSpeed: number
  /** [P0] Wind direction in radians (0–2π). 0 = +x axis. */
  windDirRad: number
  /** [P1] Visibility fraction (0–1). Display only — no obs change at P0. */
  visibility: number
  /** [P1] Temperature °C (−20–50). Display only — no obs change at P0. */
  temperatureC: number
}

/** P0: GPS denial and jamming; P2: spoofing */
export interface EWParams {
  /** [P0] GPS denial level (0–1). Adds Gaussian noise σ = level×0.2 to obs[0:2]. */
  gpsDenialLevel: number
  /** [P0] Jamming duty cycle (0–1). Each neighbor slot zeroed per-step with this probability. */
  jamDutyCycle: number
  /** [P2] Spoofing enabled. Deferred post-demo. */
  spoofingEnabled: boolean
}

/** P1: terrain knobs — display only at P0 scope */
export interface TerrainParams {
  /** [P1] Elevation roughness (0–1). Display only. */
  elevRoughness: number
  /** [P1] Urban canyon density (0–1). Display only. */
  urbanDensity: number
}

/** P1: threat counts — narrative only at P0 scope */
export interface ThreatParams {
  /** [P1] Hostile UAS count (0–10). Scenario narrative; attrition_inject_rate drives actual kills. */
  hostileUasCount: number
  /** [P1] Moving target normalized speed (0–1). Display only. */
  movingTargetSpeed: number
}

/** P1: rules of engagement */
export interface ROEParams {
  /** [P1] Engagement authority level. Display only at P0 scope. */
  engagementAuthority: 'hold-fire' | 'weapons-tight' | 'weapons-free'
  /** [P1] Minimum standoff meters (0–20). Display only. */
  minStandoffM: number
  /** [P2] Civilian density (0–1). Deferred. */
  civilianDensity: number
  /** [P0→max_steps] Time limit seconds (30–600). Maps to max_steps at env creation. */
  timeLimitSec: number
}

/** P0: swarm size, battery envelope, and attrition */
export interface LogisticsParams {
  /** [P0] Swarm size (2–12). Must match trained checkpoint — changes obs/act tensor shapes. */
  swarmSize: number
  /** [P0] Battery envelope seconds (30–600). Caps max_steps. */
  batteryEnvelopeSec: number
  /** [P0] Per-step probability of a random agent kill (0–0.5). */
  attritionInjectRate: number
}

/** Top-level battlefield configuration object. */
export interface BattlefieldParams {
  /** Scenario identifier (must match a key in scenarios.ts). */
  envId: string
  weather: WeatherParams
  ew: EWParams
  terrain: TerrainParams
  threat: ThreatParams
  roe: ROEParams
  logistics: LogisticsParams
}

// ────────────────────────────────────────────────────────────── validation ──

/** Per-field bounds used for range validation. */
export const PARAM_BOUNDS = {
  weather: {
    windSpeed:     { min: 0,   max: 15 },
    windDirRad:    { min: 0,   max: 2 * Math.PI },
    visibility:    { min: 0,   max: 1 },
    temperatureC:  { min: -20, max: 50 },
  },
  ew: {
    gpsDenialLevel: { min: 0, max: 1 },
    jamDutyCycle:   { min: 0, max: 1 },
  },
  terrain: {
    elevRoughness: { min: 0, max: 1 },
    urbanDensity:  { min: 0, max: 1 },
  },
  threat: {
    hostileUasCount:    { min: 0,  max: 10 },
    movingTargetSpeed:  { min: 0,  max: 1 },
  },
  roe: {
    minStandoffM:   { min: 0,  max: 20 },
    civilianDensity:{ min: 0,  max: 1 },
    timeLimitSec:   { min: 30, max: 600 },
  },
  logistics: {
    swarmSize:            { min: 2,  max: 12 },
    batteryEnvelopeSec:   { min: 30, max: 600 },
    attritionInjectRate:  { min: 0,  max: 0.5 },
  },
} as const

export type ValidationError = { field: string; message: string }

/**
 * Validate a BattlefieldParams object.
 * Returns an array of errors (empty if valid).
 */
export function validateBattlefieldParams(p: BattlefieldParams): ValidationError[] {
  const errors: ValidationError[] = []

  function checkRange(
    domain: string,
    field: string,
    value: number,
    min: number,
    max: number,
  ) {
    if (value < min || value > max) {
      errors.push({
        field: `${domain}.${field}`,
        message: `${domain}.${field} = ${value} is out of range [${min}, ${max}]`,
      })
    }
  }

  // weather
  checkRange('weather', 'windSpeed',    p.weather.windSpeed,    0,   15)
  checkRange('weather', 'windDirRad',   p.weather.windDirRad,   0,   2 * Math.PI)
  checkRange('weather', 'visibility',   p.weather.visibility,   0,   1)
  checkRange('weather', 'temperatureC', p.weather.temperatureC, -20, 50)

  // ew
  checkRange('ew', 'gpsDenialLevel', p.ew.gpsDenialLevel, 0, 1)
  checkRange('ew', 'jamDutyCycle',   p.ew.jamDutyCycle,   0, 1)

  // terrain
  checkRange('terrain', 'elevRoughness', p.terrain.elevRoughness, 0, 1)
  checkRange('terrain', 'urbanDensity',  p.terrain.urbanDensity,  0, 1)

  // threat
  checkRange('threat', 'hostileUasCount',   p.threat.hostileUasCount,   0,  10)
  checkRange('threat', 'movingTargetSpeed', p.threat.movingTargetSpeed, 0,  1)

  // roe
  checkRange('roe', 'minStandoffM',    p.roe.minStandoffM,    0,  20)
  checkRange('roe', 'civilianDensity', p.roe.civilianDensity, 0,  1)
  checkRange('roe', 'timeLimitSec',    p.roe.timeLimitSec,    30, 600)

  // logistics
  checkRange('logistics', 'swarmSize',           p.logistics.swarmSize,           2,   12)
  checkRange('logistics', 'batteryEnvelopeSec',  p.logistics.batteryEnvelopeSec,  30,  600)
  checkRange('logistics', 'attritionInjectRate', p.logistics.attritionInjectRate, 0,   0.5)

  return errors
}

// ────────────────────────────────────────────────────────────── defaults ──

const GARRISON_DEFAULTS: Omit<BattlefieldParams, 'envId'> = {
  weather:  { windSpeed: 0,   windDirRad: 0,       visibility: 1.0, temperatureC: 20 },
  ew:       { gpsDenialLevel: 0, jamDutyCycle: 0,  spoofingEnabled: false },
  terrain:  { elevRoughness: 0,  urbanDensity: 0 },
  threat:   { hostileUasCount: 0, movingTargetSpeed: 0.3 },
  roe:      { engagementAuthority: 'hold-fire', minStandoffM: 0, civilianDensity: 0, timeLimitSec: 400 },
  logistics:{ swarmSize: 5, batteryEnvelopeSec: 400, attritionInjectRate: 0 },
}

/**
 * Per-scenario combat-stress defaults.
 * These match the scenario matrix in docs/battlefield-parameters.md §3.
 */
export const SCENARIO_DEFAULTS: Record<string, BattlefieldParams> = {
  'drone-vs-drone': {
    envId: 'drone-vs-drone',
    weather:  { windSpeed: 3.0, windDirRad: Math.PI / 6, visibility: 1.0, temperatureC: 20 },
    ew:       { gpsDenialLevel: 0.0, jamDutyCycle: 0.2, spoofingEnabled: false },
    terrain:  { elevRoughness: 0.0, urbanDensity: 0.0 },
    threat:   { hostileUasCount: 3, movingTargetSpeed: 0.5 },
    roe:      { engagementAuthority: 'weapons-tight', minStandoffM: 0, civilianDensity: 0, timeLimitSec: 320 },
    logistics:{ swarmSize: 6, batteryEnvelopeSec: 320, attritionInjectRate: 0.0 },
  },
  'moving-target-track': {
    envId: 'moving-target-track',
    weather:  { windSpeed: 2.0, windDirRad: Math.PI / 3, visibility: 1.0, temperatureC: 20 },
    ew:       { gpsDenialLevel: 0.0, jamDutyCycle: 0.0, spoofingEnabled: false },
    terrain:  { elevRoughness: 0.0, urbanDensity: 0.0 },
    threat:   { hostileUasCount: 0, movingTargetSpeed: 0.8 },
    roe:      { engagementAuthority: 'weapons-tight', minStandoffM: 0, civilianDensity: 0, timeLimitSec: 300 },
    logistics:{ swarmSize: 4, batteryEnvelopeSec: 300, attritionInjectRate: 0.02 },
  },
  'search-and-interdict': {
    envId: 'search-and-interdict',
    weather:  { windSpeed: 4.0, windDirRad: Math.PI / 4, visibility: 0.6, temperatureC: 15 },
    ew:       { gpsDenialLevel: 0.7, jamDutyCycle: 0.4, spoofingEnabled: false },
    terrain:  { elevRoughness: 0.0, urbanDensity: 0.0 },
    threat:   { hostileUasCount: 1, movingTargetSpeed: 0.7 },
    roe:      { engagementAuthority: 'weapons-tight', minStandoffM: 0, civilianDensity: 0, timeLimitSec: 360 },
    logistics:{ swarmSize: 5, batteryEnvelopeSec: 360, attritionInjectRate: 0.02 },
  },
  'defend-asset': {
    envId: 'defend-asset',
    weather:  { windSpeed: 2.0, windDirRad: Math.PI / 2, visibility: 1.0, temperatureC: 20 },
    ew:       { gpsDenialLevel: 0.0, jamDutyCycle: 0.1, spoofingEnabled: false },
    terrain:  { elevRoughness: 0.0, urbanDensity: 0.0 },
    threat:   { hostileUasCount: 4, movingTargetSpeed: 0.5 },
    roe:      { engagementAuthority: 'weapons-tight', minStandoffM: 5, civilianDensity: 0, timeLimitSec: 280 },
    logistics:{ swarmSize: 5, batteryEnvelopeSec: 280, attritionInjectRate: 0.05 },
  },
  'swarm-vs-swarm-race': {
    envId: 'swarm-vs-swarm-race',
    weather:  { windSpeed: 5.0, windDirRad: Math.PI / 4, visibility: 0.8, temperatureC: 10 },
    ew:       { gpsDenialLevel: 0.5, jamDutyCycle: 0.4, spoofingEnabled: false },
    terrain:  { elevRoughness: 0.0, urbanDensity: 0.0 },
    threat:   { hostileUasCount: 6, movingTargetSpeed: 0.6 },
    roe:      { engagementAuthority: 'weapons-free', minStandoffM: 0, civilianDensity: 0, timeLimitSec: 320 },
    logistics:{ swarmSize: 6, batteryEnvelopeSec: 320, attritionInjectRate: 0.04 },
  },
}

/**
 * Return the scenario's combat-stress defaults, falling back to garrison defaults
 * for any scenario not in the registry.
 */
export function getScenarioDefaults(envId: string): BattlefieldParams {
  return SCENARIO_DEFAULTS[envId] ?? { ...GARRISON_DEFAULTS, envId }
}
