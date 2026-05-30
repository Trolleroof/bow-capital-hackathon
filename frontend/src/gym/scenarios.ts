export type ScenarioStatus = 'Ready'

export interface ScenarioTelemetry {
  label: string
  value: string
}

export interface ScenarioCard {
  id: string
  name: string
  label: string
  summary: string
  status: ScenarioStatus
  intent: string
  trainingHook: string
  observation: string
  action: string
  reward: string
  notes: string[]
  telemetryLabels: [string, string, string]
}

export const scenarios: ScenarioCard[] = [
  {
    id: 'drone-vs-drone',
    name: 'Drone vs Drone',
    label: 'Area denial',
    summary: 'Two teams contest the same airspace with elimination pressure near the center lane.',
    status: 'Ready',
    intent: 'Teach spacing, pressure, and survival when two swarms collide inside a jammed battlespace.',
    trainingHook: 'make_scenario_env("drone-vs-drone")',
    observation: 'Local neighbors, threat lane occupancy, obstacle bearings, friendly/alive counts.',
    action: '2D velocity command per drone; optional fire-control gate stays scripted for demos.',
    reward: 'Score for lane control and surviving contacts; penalties for blue-on-blue crowding and losses.',
    notes: [
      'Use as the aggression and deconfliction bring-up case.',
      'Best for the kill-an-agent recovery demo and contested-airspace visuals.',
    ],
    telemetryLabels: ['Blue alive', 'Red alive', 'Control'],
  },
  {
    id: 'moving-target-track',
    name: 'Moving Target Track',
    label: 'Custody',
    summary: 'Ground vehicles weave through blind spots while the swarm tries to hold visual custody.',
    status: 'Ready',
    intent: 'Train handoff and shadowing behavior without dropping track on evasive movers.',
    trainingHook: 'make_scenario_env("moving-target-track")',
    observation: 'Target-relative bearings, occlusion mask, nearest wingman offsets, track-confidence bins.',
    action: '2D velocity command with altitude held constant in the point-mass sim.',
    reward: 'Reward continuous custody and multi-angle coverage; penalize lost track and obstacle clipping.',
    notes: [
      'Maps directly to operator tasking around escort, surveillance, and reacquisition.',
      'Good first bridge from the coverage env to target-centric RL objectives.',
    ],
    telemetryLabels: ['Targets tracked', 'Occlusions', 'Custody'],
  },
  {
    id: 'search-and-interdict',
    name: 'Search & Interdict',
    label: 'GPS denied',
    summary: 'Sweep a cluttered indoor floor, find the hidden mover, and collapse the net before it escapes.',
    status: 'Ready',
    intent: 'Exercise search under uncertainty, then converge once contact is made.',
    trainingHook: 'make_scenario_env("search-and-interdict")',
    observation: 'Coverage patch, jammer pockets, obstacle map slices, last-seen target breadcrumb.',
    action: '2D velocity command with decentralized local observations only.',
    reward: 'New search coverage before contact, then fast convergence and perimeter closure after contact.',
    notes: [
      'Closest descendant of the existing land-coverage environment.',
      'Useful as the default env for policy bring-up because it preserves exploration pressure.',
    ],
    telemetryLabels: ['Cells swept', 'Threat lock', 'Intercept ETA'],
  },
  {
    id: 'defend-asset',
    name: 'Defend Asset',
    label: 'Shield ring',
    summary: 'A fixed asset sits at center while inbound agents probe from multiple approach vectors.',
    status: 'Ready',
    intent: 'Train persistent perimeter defense and rapid interception before the asset is breached.',
    trainingHook: 'make_scenario_env("defend-asset")',
    observation: 'Asset-relative bearings, interceptor gaps, inbound velocities, defended-sector occupancy.',
    action: '2D velocity command around a fixed objective with shared team reward.',
    reward: 'Reward keeping hostiles outside the ring and intercepting early; penalize breaches and idle sectors.',
    notes: [
      'Clean operator story: one thing to protect, clear win/loss state, clear sector telemetry.',
      'Also works as a benchmark for role specialization once non-scout roles exist.',
    ],
    telemetryLabels: ['Breaches', 'Shield integrity', 'Interceptors'],
  },
  {
    id: 'swarm-vs-swarm-race',
    name: 'Swarm vs Swarm Coverage Race',
    label: 'Competitive sweep',
    summary: 'Two teams race to map contested cells first while comms stay denied and lanes stay noisy.',
    status: 'Ready',
    intent: 'Stress-test exploration efficiency and jamming tolerance in a competitive scoring loop.',
    trainingHook: 'make_scenario_env("swarm-vs-swarm-race")',
    observation: 'Coverage patch, contested-cell heat, nearest rival offsets, jamming corridor proximity.',
    action: '2D velocity command with the same point-mass dynamics as the base swarm env.',
    reward: 'Points for first-touch coverage and holding contested zones; penalties for collisions and dead zones.',
    notes: [
      'Optional in the issue, but cheap to wire once the scenario registry exists.',
      'Good for demos because the scoreboard moves constantly without needing weapon semantics.',
    ],
    telemetryLabels: ['Blue score', 'Red score', 'Contested'],
  },
]

export function getScenarioById(id: string) {
  return scenarios.find((scenario) => scenario.id === id) ?? scenarios[0]
}
