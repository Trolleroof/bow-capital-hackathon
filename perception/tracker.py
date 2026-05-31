"""Norfair-based multi-object tracker with follow-mode lock."""
from __future__ import annotations

import numpy as np
print("[import/tracker] numpy ok", flush=True)

import norfair
print("[import/tracker] norfair ok", flush=True)

from norfair import Detection as NorfairDetection, Tracker

import config
from detector import Detection
print("[import/tracker] detector ok", flush=True)


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
    #
    # State machine:  Proposed --> Followed --> Confirmed
    #                              <-- R --     <-- R --
    # Invariant: Confirmed always implies Followed (_primary_id == confirmed id).

    def confirm_target(self, track_id: int) -> None:
        """Followed → Confirmed. Also locks follow so the invariant holds."""
        self._confirmed_ids = {track_id}
        self._primary_id    = track_id

    def lock_follow(self, track_id: int) -> None:
        """Proposed → Followed."""
        self._primary_id = track_id

    def release(self) -> int | None:
        """Step back one level.

        Confirmed → Followed  (clear confirmed, keep follow lock)
        Followed  → Proposed  (clear follow lock)

        Returns the confirmed track_id when a confirm is released so the caller
        can invoke gallery.release_confirm(); returns None otherwise.
        """
        if self._confirmed_ids:
            released = next(iter(self._confirmed_ids))
            self._confirmed_ids.clear()
            # _primary_id intentionally kept -- still following
            return released
        else:
            self._primary_id = None
            return None

    def unconfirm_all(self) -> None:
        """Full reset -- clears both confirmed target and follow lock."""
        self._confirmed_ids.clear()
        self._primary_id = None

    def reassign_confirmed(self, new_track_id: int) -> None:
        """ReID re-linked the confirmed target to a new Norfair track ID."""
        self._confirmed_ids = {new_track_id}
        self._primary_id    = new_track_id

    def clear_confirmed(self, track_id: int) -> None:
        self._confirmed_ids.discard(track_id)

    def refresh_flags(self, objects: list[TrackedObject]) -> None:
        """Re-apply confirmed/follow flags to an existing object list.
        Called after remote commands change tracker state mid-frame.
        """
        for obj in objects:
            obj.confirmed  = obj.id in self._confirmed_ids
            obj.is_primary = obj.id == self._primary_id


# ------------------------------------------------------------------

def _bbox_centroid(bbox: list[int]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([[x + w / 2, y + h / 2]])
