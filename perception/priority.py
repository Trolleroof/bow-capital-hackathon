"""Priority scoring + stable candidate buffer."""
from __future__ import annotations

from collections import Counter, deque

import config
from tracker import TrackedObject


def score(obj: TrackedObject) -> float:
    """Higher = higher priority."""
    weight = config.CLASS_WEIGHTS.get(obj.cls, 0.4)
    face_bonus = 1.0 if obj.has_face else 0.7
    return obj.conf * weight * face_bonus


def top_candidate(objects: list[TrackedObject]) -> TrackedObject | None:
    """Instantaneous top candidate (used internally by the buffer)."""
    candidates = [o for o in objects if not o.confirmed and not o.is_primary
                  and o.allegiance != "friend"]
    if not candidates:
        return None
    return max(candidates, key=score)


class CandidateBuffer:
    """
    Stabilises the proposed target over a rolling window of frames.

    Each frame, record the instantaneous top candidate ID.
    The buffered proposal is whichever ID has appeared most in the window --
    it won't flip until a different candidate consistently dominates.
    """

    def __init__(self, window: int | None = None) -> None:
        n = window if window is not None else config.CANDIDATE_BUFFER_FRAMES
        self._history: deque[int | None] = deque(maxlen=n)
        self._stable_id: int | None = None

    def update(self, objects: list[TrackedObject]) -> TrackedObject | None:
        instant = top_candidate(objects)
        self._history.append(instant.id if instant else None)

        counts = Counter(id_ for id_ in self._history if id_ is not None)
        if counts:
            self._stable_id = counts.most_common(1)[0][0]
        else:
            self._stable_id = None

        # Return the TrackedObject for the stable ID if it's still visible
        if self._stable_id is None:
            return None
        for obj in objects:
            if obj.id == self._stable_id and not obj.confirmed and not obj.is_primary:
                return obj
        # Stable ID dropped out of frame -- keep buffering but return nothing
        return None
