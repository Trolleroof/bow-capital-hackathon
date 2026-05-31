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
    id: 'hunt-and-seek',
    name: 'Hunt & Seek (3D)',
    label: '3D pursuit',
    summary: 'A 3D swarm searches a volumetric obstacle field, finds a slower evading target, and corners it.',
    status: 'Ready',
    observation: '3D own pos/vel, neighbor offsets, nearest volumetric obstacles, and a target block (relative offset, distance, visible-now, team-contact, last-seen age).',
    action: 'Continuous 3D velocity command per drone (x, y, z) from the exported ONNX actor.',
    reward: 'Dense closing-distance pull, first-contact bonus, then a large team capture reward when ≥2 drones box the target in. Obstacle, crowd, and bounds penalties.',
    notes: [
      'True 3D: drones fly between tall towers and over short blocks — altitude is a real tactic.',
      'The target actively evades (slower than the drones) and uses obstacles to break line-of-sight.',
    ],
    telemetryLabels: ['Captures', 'Contact', 'Min range'],
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
