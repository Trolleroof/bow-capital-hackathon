"""
perception/main.py -- CombatOS Targeting Loop

Operator flow (state machine):
  PROPOSED  --[F]--> FOLLOWED  --[C]--> CONFIRMED
            <--[R]--           <--[R]--

  F  -- follow:   lock onto proposed candidate (or re-lock on confirmed target if visible)
  C  -- confirm:  lock in the followed target (requires a followed target)
  R  -- release:  step back one level (Confirmed→Followed, Followed→Proposed)
  U  -- full reset: clear confirmed + follow + ReID gallery
  Q  -- quit
"""
from __future__ import annotations

import os
import queue
import sys
import time

# Must be set before numpy/scipy/norfair import -- prevents OpenBLAS thread-init
# deadlock on Windows (common with filterpy/scipy pulled in by norfair).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

print("[import] stdlib ok", flush=True)

import cv2
print("[import] cv2 ok", flush=True)

import numpy as np
print("[import] numpy ok", flush=True)

import config
print("[import] config ok", flush=True)

from bus import BusPublisher
print("[import] bus ok", flush=True)

from detector import Detector
print("[import] detector ok", flush=True)

from priority import CandidateBuffer
print("[import] priority ok", flush=True)

from reid import ReIDGallery
print("[import] reid ok", flush=True)

from tracker import TargetTracker
print("[import] tracker ok", flush=True)

from visualizer import draw
print("[import] visualizer ok", flush=True)


def _dbg(msg: str) -> None:
    print(f"[perception][{time.time():.3f}] {msg}", flush=True)

WINDOW = "CombatOS - Targeting"


def _drain_commands(
    publisher: BusPublisher,
    tracker: TargetTracker,
    gallery: ReIDGallery,
) -> None:
    """Process all queued dashboard commands without blocking.

    Commands arrive from the WebSocket receive loop and are batched into
    publisher.commands (SimpleQueue) between frames.  Applied in order
    so a rapid follow+confirm sequence from the UI works correctly.
    """
    while True:
        try:
            cmd = publisher.commands.get_nowait()
        except queue.Empty:
            break
        action = cmd.get("action")
        tid    = cmd.get("track_id")
        if action == "confirm" and tid is not None:
            # Enforce state machine: confirm_target sets primary too, so this is always safe
            tracker.confirm_target(tid)
            gallery.confirm(tid)
            print(f"[perception] CONFIRMED (remote) target {tid}")
        elif action == "follow" and tid is not None:
            tracker.lock_follow(tid)
            print(f"[perception] FOLLOW LOCK (remote) → {tid}")
        elif action == "unconfirm":
            tracker.unconfirm_all()
            gallery.clear()
            print("[perception] Target unconfirmed (remote)")
        elif action in ("release", "release_follow"):
            tracker.release()
            print("[perception] Released (remote)")


def _crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    return frame[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]


def main() -> None:
    source = config.VIDEO_SOURCE
    _dbg(f"opening video source: {source!r}")
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"[perception] Cannot open source: {source}")
    _dbg(f"video opened  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
         f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  "
         f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}  "
         f"frames={int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    _dbg("loading detector (YOLO) ...")
    detector = Detector()
    _dbg("detector ready")

    _dbg("loading tracker ...")
    tracker = TargetTracker()
    _dbg("tracker ready")

    buffer = CandidateBuffer()

    _dbg("loading ReID gallery (may download weights on first run) ...")
    gallery = ReIDGallery()
    _dbg("ReID gallery ready")

    _dbg("connecting to WebSocket bus ...")
    publisher = BusPublisher()
    try:
        publisher.connect()
        _dbg("connected to bus")
    except Exception as e:
        _dbg(f"bus unavailable ({e}), running in local-only mode")
        publisher = None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, w, h)

    _dbg("entering main loop")
    frame_n   = 0
    t_loop    = time.time()
    t_slow_threshold = 0.5  # log any step that takes longer than this

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        t_read = time.time()
        if not ret:
            _dbg(f"cap.read() returned False at frame {frame_n} -- end of stream or read error")
            break
        if t_read - t0 > t_slow_threshold:
            _dbg(f"frame {frame_n}: SLOW cap.read() {t_read-t0:.3f}s")

        detections = detector.run(frame)
        t_det = time.time()
        if t_det - t_read > t_slow_threshold:
            _dbg(f"frame {frame_n}: SLOW detector.run() {t_det-t_read:.3f}s  detections={len(detections)}")

        objects    = tracker.update(detections)
        active_ids = {o.id for o in objects}
        t_track = time.time()

        # Drain dashboard commands (non-blocking, batched since last frame)
        if publisher:
            _drain_commands(publisher, tracker, gallery)
            tracker.refresh_flags(objects)

        # ---- passive sampling + ReID pass ----------------------------
        confirmed_visible = any(o.confirmed for o in objects)

        for obj in objects:
            c = _crop(frame, obj.bbox)
            gallery.sample(obj.id, c)

            if not confirmed_visible and not obj.confirmed and not obj.is_primary and gallery.active:
                matched_id = gallery.match(obj.id, c)
                if matched_id is not None:
                    tracker.reassign_confirmed(obj.id)
                    gallery.reassign(obj.id)
                    break

        gallery.prune(active_ids)
        t_reid = time.time()
        if t_reid - t_track > t_slow_threshold:
            _dbg(f"frame {frame_n}: SLOW reid/sample {t_reid-t_track:.3f}s")
        # --------------------------------------------------------------

        candidate = buffer.update(objects)

        followed_obj  = next((o for o in objects if o.is_primary), None)
        confirmed_obj = next((o for o in objects if o.confirmed), None)

        if publisher:
            publisher.publish(objects)

        frame = draw(frame, objects, candidate)
        cv2.imshow(WINDOW, frame)
        t_draw = time.time()

        # Periodic summary every 30 frames
        frame_n += 1
        if frame_n % 30 == 0:
            elapsed = time.time() - t_loop
            fps = 30 / elapsed
            _dbg(f"frame {frame_n}: {fps:.1f} fps  tracks={len(objects)}  "
                 f"det={t_det-t_read:.3f}s  reid={t_reid-t_track:.3f}s  "
                 f"draw={t_draw-t_reid:.3f}s")
            t_loop = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            # Follow confirmed target if it's visible; otherwise follow proposed candidate
            target = confirmed_obj or candidate
            if target:
                tracker.lock_follow(target.id)
                print(f"[perception] FOLLOW LOCK → {target.id}")
        elif key == ord("c"):
            # Confirm requires a followed target -- C on a bare candidate is not allowed
            if followed_obj:
                tracker.confirm_target(followed_obj.id)
                gallery.confirm(followed_obj.id)
                print(f"[perception] CONFIRMED target {followed_obj.id} -- {gallery.debug_info()}")
            else:
                print("[perception] Press F first to follow a target before confirming")
        elif key == ord("u"):
            tracker.unconfirm_all()
            gallery.clear()
            print("[perception] Target unconfirmed")
        elif key == ord("r"):
            tracker.release()
            print("[perception] Released")

    cap.release()
    cv2.destroyAllWindows()
    if publisher:
        publisher.close()


if __name__ == "__main__":
    main()
