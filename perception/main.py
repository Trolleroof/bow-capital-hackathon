"""
perception/main.py -- CombatOS Targeting Loop

Controls:
  c  -- confirm the buffered proposed candidate (replaces any prior confirmation)
  u  -- unconfirm (clear confirmed target + ReID gallery)
  f  -- lock follow mode on the buffered proposed candidate
  r  -- release follow mode
  q  -- quit
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

import config
from bus import BusPublisher
from detector import Detector
from priority import CandidateBuffer
from reid import ReIDGallery
from tracker import TargetTracker
from visualizer import draw

WINDOW = "CombatOS - Targeting"


def _crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    return frame[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]


def main() -> None:
    source = config.VIDEO_SOURCE
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"[perception] Cannot open source: {source}")

    detector  = Detector()
    tracker   = TargetTracker()
    buffer    = CandidateBuffer()
    gallery   = ReIDGallery()
    publisher = BusPublisher()

    try:
        publisher.connect()
        print("[perception] Connected to bus")
    except Exception as e:
        print(f"[perception] Bus unavailable ({e}), running in local-only mode")
        publisher = None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, w, h)

    print("[perception] Running. Press q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.run(frame)
        objects    = tracker.update(detections)
        active_ids = {o.id for o in objects}

        # ---- passive sampling + ReID pass ----------------------------
        confirmed_visible = any(o.confirmed for o in objects)

        for obj in objects:
            c = _crop(frame, obj.bbox)
            # Sample every track at ~1s intervals (pre-buffer or confirmed gallery)
            gallery.sample(obj.id, c)

            # ReID matching: only when confirmed target has left the tracker
            if not confirmed_visible and not obj.confirmed and not obj.is_primary and gallery.active:
                matched_id = gallery.match(obj.id, c)
                if matched_id is not None:
                    tracker.reassign_confirmed(obj.id)
                    gallery.reassign(obj.id)
                    break

        # Drop pre-buffers for tracks that left frame (unconfirmed only)
        gallery.prune(active_ids)
        # --------------------------------------------------------------

        candidate = buffer.update(objects)

        if publisher:
            publisher.publish(objects)

        frame = draw(frame, objects, candidate)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and candidate:
            tracker.confirm_target(candidate.id)
            gallery.confirm(candidate.id)
            print(f"[perception] CONFIRMED target {candidate.id} -- {gallery.debug_info()}")
        elif key == ord("u"):
            tracker.unconfirm_all()
            gallery.clear()
            print("[perception] Target unconfirmed")
        elif key == ord("f") and candidate:
            tracker.lock_follow(candidate.id)
            print(f"[perception] FOLLOW LOCK → {candidate.id}")
        elif key == ord("r"):
            tracker.release_follow()
            print("[perception] Follow released")

    cap.release()
    cv2.destroyAllWindows()
    if publisher:
        publisher.close()


if __name__ == "__main__":
    main()
