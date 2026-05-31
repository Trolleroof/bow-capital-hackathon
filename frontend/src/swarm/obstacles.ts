/**
 * obstacles.ts — TypeScript mirror of swarm/obstacles.py
 *
 * Keep the two files in lockstep. Coordinates are world units (world spans
 * [-10, 10] in X and Y). `sx`/`sy` are XY half-extents for boxes; for cylinders
 * both fields equal the radius so the observation slot is always 4 floats.
 *
 * These are exactly the props the Python training env adds to its observation
 * and collides against, so a freshly-trained policy.onnx produces the same
 * obstacle-aware behaviour in the browser.
 */

export type ObstacleKind = 'box' | 'cylinder'

export interface Obstacle {
  kind: ObstacleKind
  cx: number
  cy: number
  sx: number
  sy: number
  zCenter: number
  zExtent: number
}

const box = (
  cx: number,
  cy: number,
  sx: number,
  sy: number,
  z = 0.9,
  h = 0.9,
): Obstacle => ({ kind: 'box', cx, cy, sx, sy, zCenter: z, zExtent: h })

const cyl = (cx: number, cy: number, r: number, z = 1.0, h = 1.0): Obstacle => ({
  kind: 'cylinder',
  cx,
  cy,
  sx: r,
  sy: r,
  zCenter: z,
  zExtent: h,
})

export const SCENARIO_OBSTACLES: Record<string, Obstacle[]> = {
  'drone-vs-drone': [
    box(-3.0, 0.0, 0.5, 3.0, 0.9, 0.9),
    box(3.0, 0.0, 0.5, 3.0, 0.9, 0.9),
    cyl(0.0, 5.2, 0.35, 1.1, 2.2),
  ],
  'moving-target-track': [
    box(-4.2, 3.2, 1.4, 2.6, 1.3, 1.3),
    box(3.6, -3.4, 1.6, 2.4, 1.3, 1.3),
    box(4.4, 3.0, 1.6, 0.7, 0.5, 0.5),
  ],
  'search-and-interdict': [
    box(-4.5, -3.6, 0.9, 0.9, 0.7, 0.7),
    box(-1.6, 3.1, 0.9, 0.9, 0.7, 0.7),
    box(3.1, 1.2, 0.9, 0.9, 0.7, 0.7),
    box(4.7, -3.7, 0.9, 0.9, 0.7, 0.7),
    cyl(1.0, -0.4, 1.1, 0.8, 1.6),
  ],
  'defend-asset': [
    cyl(0.0, 0.0, 1.0, 0.2, 0.4),
    box(0.0, 4.6, 2.0, 0.6, 0.5, 0.5),
    box(0.0, -4.6, 2.0, 0.6, 0.5, 0.5),
  ],
  'navigate-to-target': [
    box(-4.0, 2.5, 0.5, 1.2, 0.6, 1.2),
    box(-4.0, -2.5, 0.5, 1.2, 0.6, 1.2),
    cyl(-1.5, 1.8, 0.7, 0.75, 1.5),
    cyl(-1.5, -1.8, 0.7, 0.75, 1.5),
    box(1.5, 3.2, 0.5, 1.0, 0.5, 1.0),
    box(1.5, -3.2, 0.5, 1.0, 0.5, 1.0),
    cyl(4.2, 0.0, 0.8, 0.9, 1.8),
  ],
}

export function obstaclesFor(scenarioId: string | null | undefined): Obstacle[] {
  if (!scenarioId) return []
  return SCENARIO_OBSTACLES[scenarioId] ?? []
}
