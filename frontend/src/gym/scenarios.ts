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
    observation: 'Local neighbors, obstacles, nearest hostile offsets, hostile count, and post-kill orbit cues.',
    action: '2D velocity command per drone from the exported ONNX actor.',
    reward: 'Reward hostile elimination, approach pressure, post-kill orbit discipline, and deconfliction.',
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
    observation: 'Target-relative position/velocity, custody flags, target distance, and angular slot error.',
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
    observation: 'Coverage patch, obstacles, search/contact/intercept phase, last-seen target cue, and intercept distance.',
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
    observation: 'Asset bearing, nearest inbound hostile position/velocity, breach risk, ring error, and sector error.',
    action: '2D velocity command around a fixed objective with shared team reward.',
    reward: 'Reward keeping hostiles outside the ring and intercepting early; penalize breaches and idle sectors.',
    notes: [
      'Clean operator story: one thing to protect, clear win/loss state, clear sector telemetry.',
      'Also works as a benchmark for role specialization once non-scout roles exist.',
    ],
    telemetryLabels: ['Breaches', 'Shield integrity', 'Interceptors'],
  },
  {
    id: 'navigate-to-target',
    name: 'Navigate to Target',
    label: 'Obstacle avoidance',
    summary: 'A single drone must thread through a cluttered corridor and reach the goal zone at the far end.',
    status: 'Ready',
    observation: 'Own position/velocity, nearest obstacle offsets and extents, target bearing, and distance.',
    action: '2D velocity command for one drone; no teammates, no hostiles.',
    reward: 'Dense approach reward scales with proximity to goal; large one-time bonus on reaching the target zone.',
    notes: [
      'Cleanest bring-up case for obstacle avoidance — one agent, one goal, no team coordination required.',
      'Good for verifying that the obstacle observation slots and collision push-out are wired correctly.',
    ],
    telemetryLabels: ['Distance', 'Obstacles hit', 'Progress'],
  },
]

export function getScenarioById(id: string) {
  return scenarios.find((scenario) => scenario.id === id) ?? scenarios[0]
}
