/**
 * Mirrors pybullet_swarm_video/pybullet_swarm_video/config.py.
 * Retrieved with:
 * uv run --project pybullet_swarm_video python -c "..."
 */

export const PYBULLET_DEMO_CONFIG = {
  simulation: {
    numDrones: 5,
    numTroops: 6,
    durationSec: 12,
    timeStep: 1 / 30,
    worldHalfExtentM: 30,
    droneAltitudeM: 12,
    droneSpeedMps: 5,
    droneRingRadiusM: 9,
    droneSeparationGain: 2.4,
    troopSpacingM: 1.6,
    troopStrideMps: 1,
    camera: {
      width: 640,
      height: 360,
      fovDeg: 78,
      near: 0.05,
      far: 120,
      tiltDeg: 58,
      forwardOffsetM: 0.18,
    },
  },
  recording: {
    outputPath: 'output/drone_spy_demo.mp4',
    fps: 12,
  },
  orchestrator: {
    controlWsUrl: 'ws://localhost:8000',
    imageWsUrl: 'ws://localhost:8001',
    rawTopic: 'drone_fpv_raw',
    hudTopic: 'drone_fpv_hud',
    stateTopic: 'drone_fpv_state',
    detectionsTopic: 'drone_detections',
    dashboardRawTopic: 'fpv_raw',
    dashboardHudTopic: 'fpv_hud',
    dashboardDroneId: 0,
  },
} as const
