"""Norfair-based multi-object tracker with follow-mode lock."""
from __future__ import annotations

import numpy as np
import norfair
from norfair import Detection as NorfairDetection, Tracker

import config
from detector import Detection


class TrackedObject:
    def __init__(self, track_id: int, det: Detection) -> None:
        self.id = track_id
        self.cls = det.cls
        self.conf = det.conf
        self.bbox = det.bbox
        self.has_face = det.has_face
        self.confirmed = False   # operator-confirmed target
        self.is_primary = False  # follow-mode lock


class TargetTracker:
    def __init__(self) -> None:
        self._tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=config.MAX_DISTANCE,
            hit_counter_max=config.MAX_LOST_FRAMES,
        )
        self._primary_id: int | None = None   # follow-mode locked ID
        self._confirmed_ids: set[int] = set() # operator-confirmed targets

    # ------------------------------------------------------------------
    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        norfair_dets = [
            NorfairDetection(
                points=_bbox_centroid(d.bbox),
                scores=np.array([d.conf]),
                label=d.cls,
                data=d,
            )
            for d in detections
        ]
        tracked = self._tracker.update(norfair_dets)

        objects: list[TrackedObject] = []
        for t in tracked:
            src_det: Detection = t.last_detection.data if t.last_detection else None
            if src_det is None:
                continue
            obj = TrackedObject(t.id, src_det)
            obj.confirmed = t.id in self._confirmed_ids
            obj.is_primary = t.id == self._primary_id
            objects.append(obj)

        return objects

    # ------------------------------------------------------------------
    # Operator actions (called from the bus or UI)

    def confirm_target(self, track_id: int) -> None:
        """Operator confirms a proposed target."""
        self._confirmed_ids.add(track_id)

    def lock_follow(self, track_id: int) -> None:
        """Enter follow mode on a specific track."""
        self._primary_id = track_id

    def release_follow(self) -> None:
        self._primary_id = None

    def clear_confirmed(self, track_id: int) -> None:
        self._confirmed_ids.discard(track_id)


# ------------------------------------------------------------------

def _bbox_centroid(bbox: list[int]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([[x + w / 2, y + h / 2]])
