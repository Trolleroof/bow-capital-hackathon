"""OpenCV debug overlay -- draw boxes, labels, and target lock on a frame."""
from __future__ import annotations

import cv2
import numpy as np

from tracker import TrackedObject

# Colors (BGR)
_COLOR = {
    "default":   (180, 180, 180),
    "troop":     (0, 200, 255),
    "vehicle":   (0, 128, 255),
    "ugv":       (255, 165, 0),
    "aerial":    (255, 255, 0),
    "confirmed": (0, 255, 0),
    "primary":   (0, 0, 255),
    "candidate": (255, 0, 255),
}


def draw(
    frame: np.ndarray,
    objects: list[TrackedObject],
    candidate: TrackedObject | None,
) -> np.ndarray:
    for obj in objects:
        x, y, w, h = obj.bbox
        color = _pick_color(obj, candidate)
        thickness = 3 if (obj.is_primary or obj.confirmed) else 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        label = f"[{obj.id}] {obj.cls.upper()} {obj.conf:.2f}"
        if obj.has_face:
            label += " [FACE]"
        if obj.confirmed:
            label += " CONFIRMED"
        elif obj.is_primary:
            label += " LOCKED"
        elif candidate and obj.id == candidate.id:
            label += " PROPOSED"

        cv2.putText(
            frame, label, (x, max(y - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    # HUD
    _draw_hud(frame, objects, candidate)
    return frame


def _pick_color(obj: TrackedObject, candidate: TrackedObject | None) -> tuple[int, int, int]:
    if obj.confirmed:
        return _COLOR["confirmed"]
    if obj.is_primary:
        return _COLOR["primary"]
    if candidate and obj.id == candidate.id:
        return _COLOR["candidate"]
    return _COLOR.get(obj.cls, _COLOR["default"])


def _draw_hud(
    frame: np.ndarray,
    objects: list[TrackedObject],
    candidate: TrackedObject | None,
) -> None:
    h, w = frame.shape[:2]
    lines = [
        f"TRACKS: {len(objects)}",
        f"CONFIRMED: {sum(1 for o in objects if o.confirmed)}",
        f"CANDIDATE: {candidate.id if candidate else 'none'}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (10, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
