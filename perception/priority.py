"""Priority scoring — selects the best candidate to propose to the operator."""
from __future__ import annotations

import config
from tracker import TrackedObject


def score(obj: TrackedObject) -> float:
    """Higher = higher priority."""
    weight = config.CLASS_WEIGHTS.get(obj.cls, 0.4)
    face_bonus = 1.0 if obj.has_face else 0.7
    return obj.conf * weight * face_bonus


def top_candidate(objects: list[TrackedObject]) -> TrackedObject | None:
    """Return the highest-priority unconfirmed object."""
    candidates = [o for o in objects if not o.confirmed and not o.is_primary]
    if not candidates:
        return None
    return max(candidates, key=score)
