"""Per-scenario 3D obstacle layouts shared by training env and PyBullet renderer.

The training env (`SwarmEnv`) projects each obstacle to its XY footprint and
performs hard collision push-out + adds the nearest-K obstacles to the local
observation, so the policy actually learns to navigate them. The PyBullet
renderer reads from the same source, so what you see is what was trained.

Coordinates are in world units (world spans [-10, 10] in X and Y). All sizes
are HALF-extents for boxes (so a 1.0 half-extent is a 2.0-wide wall).

Tall enough to matter (block a drone flying at altitude ≈ 2 m) is the rule for
what lives here. Decorative low props (score gates, control discs) stay in the
renderer as visual-only.

Mirrors the geometry hand-coded in `swarm/pybullet_renderer.py::_build_world`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ObstacleKind = Literal["box", "cylinder"]


@dataclass(frozen=True)
class Obstacle:
    """A collidable scenery primitive.

    For ``kind="box"``  : (sx, sy) are XY half-extents (axis-aligned).
    For ``kind="cylinder"``: sx == sy == radius (we keep two fields so the
    observation slot is a uniform 4 floats per obstacle).
    """

    kind: ObstacleKind
    cx: float
    cy: float
    sx: float
    sy: float
    z_center: float = 1.0
    z_extent: float = 1.0

    def half_extent(self) -> float:
        """Larger XY footprint half-extent; used for picking nearest obstacles."""
        return max(self.sx, self.sy)


def _box(cx: float, cy: float, sx: float, sy: float, *, z: float = 0.9, h: float = 0.9) -> Obstacle:
    return Obstacle("box", cx, cy, sx, sy, z, h)


def _cyl(cx: float, cy: float, r: float, *, z: float = 1.0, h: float = 1.0) -> Obstacle:
    return Obstacle("cylinder", cx, cy, r, r, z, h)


SCENARIO_OBSTACLES: dict[str, list[Obstacle]] = {
    "drone-vs-drone": [
        _box(-3.0, 0.0, 0.5, 3.0, h=0.9),   # west blast wall
        _box( 3.0, 0.0, 0.5, 3.0, h=0.9),   # east blast wall
        _cyl( 0.0, 5.2, 0.35, h=2.2, z=1.1),  # radar mast
    ],
    "moving-target-track": [
        _box(-4.2,  3.2, 1.4, 2.6, h=1.3, z=1.3),  # warehouse A
        _box( 3.6, -3.4, 1.6, 2.4, h=1.3, z=1.3),  # warehouse B
        _box( 4.4,  3.0, 1.6, 0.7, h=0.5, z=0.5),  # fuel truck
    ],
    "search-and-interdict": [
        _box(-4.5, -3.6, 0.9, 0.9, h=0.7, z=0.7),
        _box(-1.6,  3.1, 0.9, 0.9, h=0.7, z=0.7),
        _box( 3.1,  1.2, 0.9, 0.9, h=0.7, z=0.7),
        _box( 4.7, -3.7, 0.9, 0.9, h=0.7, z=0.7),
        _cyl( 1.0, -0.4, 1.1, h=1.6, z=0.8),  # jammer pillar
    ],
    "defend-asset": [
        _cyl(0.0,  0.0, 1.0, h=0.4, z=0.2),    # asset itself (drones avoid the pedestal)
        _box(0.0,  4.6, 2.0, 0.6, h=0.5, z=0.5),  # hardpoint north
        _box(0.0, -4.6, 2.0, 0.6, h=0.5, z=0.5),  # hardpoint south
    ],
    "navigate-to-target": [
        # Corridor obstacles the single drone must weave through left→right.
        _box(-4.0,  2.5, 0.5, 1.2, h=1.2, z=0.6),   # obstacle row 1 top
        _box(-4.0, -2.5, 0.5, 1.2, h=1.2, z=0.6),   # obstacle row 1 bottom
        _cyl(-1.5,  1.8, 0.7, h=1.5, z=0.75),         # pillar cluster
        _cyl(-1.5, -1.8, 0.7, h=1.5, z=0.75),
        _box( 1.5,  3.2, 0.5, 1.0, h=1.0, z=0.5),   # mid-field wall
        _box( 1.5, -3.2, 0.5, 1.0, h=1.0, z=0.5),
        _cyl( 4.2,  0.0, 0.8, h=1.8, z=0.9),          # choke-point pillar near goal
    ],
}


def obstacles_for(scenario_id: str) -> list[Obstacle]:
    """Return the obstacle list for a scenario, empty for unknown scenarios."""
    return list(SCENARIO_OBSTACLES.get(scenario_id, ()))


def as_dict(obstacle: Obstacle) -> dict:
    """JSON-friendly serialization (for snapshots / debugging / TS mirroring)."""
    return {
        "kind": obstacle.kind,
        "cx": obstacle.cx,
        "cy": obstacle.cy,
        "sx": obstacle.sx,
        "sy": obstacle.sy,
        "z_center": obstacle.z_center,
        "z_extent": obstacle.z_extent,
    }
