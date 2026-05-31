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
        self.allegiance: str | None = det.allegiance  # "friend" | "foe" | None
        self.confirmed = False   # operator-confirmed target
        self.is_primary = False  # follow-mode lock


class TargetTracker:
    def __init__(self) -> None:
        self._tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=config.MAX_DISTANCE,
            hit_counter_max=config.MAX_LOST_FRAMES,
            initialization_delay=config.TRACK_INIT_DELAY,
        )
        self._primary_id: int | None = None   # follow-mode locked ID
        self._confirmed_ids: set[int] = set() # operator-confirmed targets
        self._smooth: dict[int, list[float]] = {}  # track_id → smoothed bbox
        self._allegiance: dict[int, str | None] = {}  # track_id → allegiance

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

        # Drop tracks that haven't yet cleared the initialization window
        active_ids: set[int] = set()
        objects: list[TrackedObject] = []
        for t in tracked:
            if getattr(t, 'is_initializing', False):
                continue
            src_det: Detection = t.last_detection.data if t.last_detection else None
            if src_det is None:
                continue

            # EMA bbox smoothing to reduce per-frame jitter
            alpha = config.BBOX_SMOOTH_ALPHA
            raw = list(src_det.bbox)
            if t.id in self._smooth and alpha > 0:
                prev = self._smooth[t.id]
                smoothed = [alpha * r + (1 - alpha) * p for r, p in zip(raw, prev)]
            else:
                smoothed = raw
            self._smooth[t.id] = smoothed
            src_det.bbox = [int(v) for v in smoothed]

            active_ids.add(t.id)
            self._allegiance[t.id] = src_det.allegiance
            obj = TrackedObject(t.id, src_det)
            obj.confirmed = t.id in self._confirmed_ids
            obj.is_primary = t.id == self._primary_id
            objects.append(obj)

        # Prune stale smooth and allegiance entries
        gone = set(self._smooth) - active_ids
        for tid in gone:
            del self._smooth[tid]
            self._allegiance.pop(tid, None)

        return objects

    # ------------------------------------------------------------------
    # Operator actions (called from the bus or UI)
    #
    # State machine:  Proposed --> Followed --> Confirmed
    #                              <-- R --     <-- R --
    # Invariant: Confirmed always implies Followed (_primary_id == confirmed id).

    def confirm_target(self, track_id: int) -> bool:
        """Followed → Confirmed. Returns False (no-op) if the track is a friend."""
        if self._allegiance.get(track_id) == "friend":
            print(f"[tracker] IFF: track {track_id} is FRIEND -- confirm blocked")
            return False
        self._confirmed_ids = {track_id}
        self._primary_id    = track_id
        return True

    def lock_follow(self, track_id: int) -> bool:
        """Proposed → Followed. Returns False (no-op) if the track is a friend."""
        if self._allegiance.get(track_id) == "friend":
            print(f"[tracker] IFF: track {track_id} is FRIEND -- follow blocked")
            return False
        self._primary_id = track_id
        return True

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
