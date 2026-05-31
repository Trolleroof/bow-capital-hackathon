export type CommandPanelId =
  | 'telemetry'
  | 'vslam'
  | 'keyframe'
  | 'targeting'
  | 'detections'
  | 'swarm-a'
  | 'swarm-b'

export const PANEL_LABELS: Record<CommandPanelId, string> = {
  telemetry: 'FIELD TELEMETRY',
  vslam: 'NAVIGATION · STEREO VSLAM',
  keyframe: 'SLAM KEYFRAME',
  targeting: 'TARGETING FEED',
  detections: 'DETECTIONS',
  'swarm-a': 'SWARM · SECTOR A',
  'swarm-b': 'SWARM · SECTOR B',
}
